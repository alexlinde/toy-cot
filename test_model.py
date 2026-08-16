"""
Interactive Tkinter GUI for the Toy VLM.

Generates random multi-shape RGB scenes, lets you ask questions about them,
and displays both the model's chain-of-thought rationale and its final
answer, alongside the scene's ground-truth object metadata.
"""

import argparse
import os
import random
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import torch
from PIL import Image, ImageTk

from model import ToyVLM, generate_response
from shapes import MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator, grid_col, grid_row
from text import TextProcessor

CANVAS_SCALE = 4  # 64x64 scene -> 256x256 display

EXAMPLE_QUESTIONS = [
    "is there a red circle",
    "how many circles are there",
    "is a square above a circle",
    "are there more squares than circles",
    "is there a triangle on the left",
    "are there any large shapes",
]


def best_device() -> torch.device:
    """Pick the best available device: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def get_model_stats(model: ToyVLM) -> dict:
    """Component-wise parameter counts and architecture summary for ToyVLM."""
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_token_encoder.parameters())
    aux_params = sum(p.numel() for p in model.auxiliary_heads.parameters())
    text_embed_params = (sum(p.numel() for p in model.token_embedding.parameters()) +
                          sum(p.numel() for p in model.position_embedding.parameters()))
    transformer_params = sum(p.numel() for p in model.transformer_blocks.parameters())
    output_params = sum(p.numel() for p in model.output_projection.parameters())

    return {
        'total_params': total_params,
        'vision_params': vision_params,
        'aux_params': aux_params,
        'text_embed_params': text_embed_params,
        'transformer_params': transformer_params,
        'output_params': output_params,
        'vocab_size': model.output_projection.out_features,
        'hidden_dim': model.token_embedding.embedding_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': getattr(model.transformer_blocks[0].attn, 'num_heads', None),
        'device': str(next(model.parameters()).device),
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def format_model_stats(stats: dict) -> str:
    return (
        f"Model statistics:\n"
        f"  Total parameters: {format_number(stats['total_params'])}\n"
        f"  Vision encoder: {format_number(stats['vision_params'])}\n"
        f"  Auxiliary heads: {format_number(stats['aux_params'])}\n"
        f"  Text embeddings: {format_number(stats['text_embed_params'])}\n"
        f"  Transformer blocks: {format_number(stats['transformer_params'])}\n"
        f"  Output layer: {format_number(stats['output_params'])}\n"
        f"  Model size: {stats['model_size_mb']:.1f} MB\n"
        f"  Architecture: {stats['hidden_dim']}d, {stats['num_layers']} layers, {stats['num_heads']} heads\n"
        f"  Vocabulary size: {format_number(stats['vocab_size'])}\n"
        f"  Device: {stats['device']}"
    )


class ToyVLMGUI:
    """Tkinter GUI for the Toy VLM: scene viewer + question box + inference."""

    def __init__(self, checkpoint: str, vocab: str):
        if not os.path.isfile(vocab):
            sys.exit(f"Vocab file not found: '{vocab}'. Train a model first with train_model.py.")
        if not os.path.isfile(checkpoint):
            sys.exit(f"Checkpoint not found: '{checkpoint}'. Train a model first with train_model.py.")

        # Build text processor + tokenizer from the saved vocabulary.
        self.text_processor = TextProcessor()
        self.text_processor.tokenizer.load_vocab(vocab)

        # Build the model and load trained weights.
        self.model = ToyVLM(self.text_processor)
        state = torch.load(checkpoint, map_location='cpu')
        self.model.load_state_dict(state)

        self.device = best_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.shape_generator = ShapeGenerator()
        self.current_image = None  # (64, 64, 3) uint8 RGB
        self.current_metadata = None

        self.question_history = []
        self.history_index = -1

        self.root = tk.Tk()
        self.root.title("Toy Vision-Language Model")
        self.root.geometry("900x560")
        self.setup_gui()

        self.generate_new_scene()

    def setup_gui(self):
        """Set up the GUI layout."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: scene image, new-scene button, ground-truth metadata.
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.canvas = tk.Canvas(
            left_frame, width=64 * CANVAS_SCALE, height=64 * CANVAS_SCALE,
            bg='black', highlightthickness=1
        )
        self.canvas.pack(pady=10)

        ttk.Button(left_frame, text="New Scene", command=self.generate_new_scene).pack(pady=(0, 10))

        ttk.Label(left_frame, text="Ground-truth objects:").pack(anchor='w')
        self.metadata_display = scrolledtext.ScrolledText(
            left_frame, height=14, width=34, wrap=tk.WORD, state='disabled'
        )
        self.metadata_display.pack(fill=tk.BOTH, expand=False)

        # Right panel: chat history + question entry.
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(right_frame, height=22, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_frame, text="Ask a question:").pack(anchor='w')

        entry_button_frame = ttk.Frame(input_frame)
        entry_button_frame.pack(fill=tk.X, pady=(5, 10))

        self.question_entry = ttk.Entry(entry_button_frame)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.question_entry.bind('<Return>', self.on_enter_pressed)
        self.question_entry.bind('<Up>', self.on_up_key)
        self.question_entry.bind('<Down>', self.on_down_key)

        ttk.Button(entry_button_frame, text="Ask Question", command=self.ask_question).pack(side=tk.RIGHT)

        # Display model statistics.
        self.display_model_stats()

        # Welcome message with real, tokenizable example questions.
        welcome_msg = "Chain-of-Thought VLM ready. Ask questions about the shapes, e.g.:\n"
        welcome_msg += "\n".join(f"  - {q}" for q in EXAMPLE_QUESTIONS)
        self.add_to_chat(welcome_msg, "System")

        self.question_entry.focus_set()

    def display_model_stats(self):
        """Display model statistics in the chat and print them at startup."""
        stats = get_model_stats(self.model)
        stats_text = format_model_stats(stats)
        print(stats_text)
        self.add_to_chat(stats_text, "System")

    def add_to_chat(self, message, sender="User"):
        """Add a message to the chat display."""
        self.chat_display.config(state='normal')
        if sender == "System":
            self.chat_display.insert(tk.END, f"[System] {message}\n\n")
        elif sender == "User":
            self.chat_display.insert(tk.END, f"[You] {message}\n")
        else:  # model response
            self.chat_display.insert(tk.END, f"[Model] {message}\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def generate_new_scene(self):
        """Generate a new multi-shape RGB scene and update the display."""
        num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        self.current_image, self.current_metadata = self.shape_generator.generate_multi_shape_image(
            num_shapes, False
        )
        self.update_canvas_display()
        self.update_metadata_display()

    def update_canvas_display(self):
        """Update the canvas with the current RGB scene, scaled up."""
        pil_img = Image.fromarray(self.current_image, 'RGB')
        pil_img = pil_img.resize((64 * CANVAS_SCALE, 64 * CANVAS_SCALE), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def update_metadata_display(self):
        """Show ground-truth shape/color/size/cell info for each object."""
        self.metadata_display.config(state='normal')
        self.metadata_display.delete('1.0', tk.END)
        for i, m in enumerate(self.current_metadata, start=1):
            row, col = grid_row(m['cy']), grid_col(m['cx'])
            self.metadata_display.insert(
                tk.END, f"{i}. {m['size_category']} {m['color']} {m['shape']} at row {row} col {col}\n"
            )
        self.metadata_display.config(state='disabled')

    def on_enter_pressed(self, event):
        """Handle Enter key press in question entry."""
        self.ask_question()

    def on_up_key(self, event):
        """Navigate to the previous question in history."""
        if not self.question_history:
            return
        if self.history_index == -1:
            self.history_index = len(self.question_history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        if 0 <= self.history_index < len(self.question_history):
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])

    def on_down_key(self, event):
        """Navigate to the next question in history."""
        if not self.question_history or self.history_index == -1:
            return
        if self.history_index < len(self.question_history) - 1:
            self.history_index += 1
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])
        else:
            self.history_index = -1
            self.question_entry.delete(0, tk.END)

    def ask_question(self):
        """Process a question about the current scene."""
        question = self.question_entry.get().strip()
        if not question:
            return

        self.question_entry.delete(0, tk.END)
        self.question_history.append(question)
        self.history_index = -1

        self.add_to_chat(question, "User")

        # Run inference in a background thread to avoid freezing the GUI.
        threading.Thread(target=self._process_question, args=(question,), daemon=True).start()

    def _process_question(self, question):
        """Run the model on the current scene and display rationale + answer."""
        image = torch.tensor(self.current_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        rationale, answer = generate_response(self.model, image, question)

        response = f"Reasoning: {rationale}\nAnswer: {answer}"
        self.root.after(0, self.add_to_chat, response, "Model")

    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive GUI for the Toy VLM.")
    parser.add_argument('--checkpoint', type=str, default='toy_vlm_cot.pth')
    parser.add_argument('--vocab', type=str, default='tokenizer_vocab.json')
    return parser.parse_args()


def main():
    args = parse_args()
    gui = ToyVLMGUI(checkpoint=args.checkpoint, vocab=args.vocab)
    gui.run()


if __name__ == "__main__":
    main()
