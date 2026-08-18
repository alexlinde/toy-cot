"""
Interactive Tkinter GUI for the Toy VLM.

Generates random multi-shape RGB scenes, lets you ask questions about them,
and displays both the model's chain-of-thought rationale and its final
answer.
"""

import argparse
import os
import random
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from model import ToyVLM, generate_response, read_aux_counts
from shapes import MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator
from text import TextProcessor

CANVAS_SCALE = 4  # 64x64 scene -> 256x256 display

# A mix of the canonical templates and the alternative phrasings each question
# type is also trained on (see questions.phrase), so the examples advertise that
# there is more than one way to ask.
EXAMPLE_QUESTIONS = [
    "is there a red circle",
    "how many circles are there",
    "is there a square above a circle",
    "is the number of squares greater than the number of circles",
    "are there any triangles on the left",
    "which shape is second from the left",
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


def make_seeded_scene(shape_generator: ShapeGenerator, seed: int = None):
    """Seed the RNGs and generate a scene, so it can be reproduced from the seed alone.

    If `seed` is None, one is drawn first (so it can still be reported back to the
    caller). Seeding happens immediately before the object-count draw and the image
    generation call, so the same seed always yields the same image and metadata.

    Returns (seed, image, metadata).
    """
    if seed is None:
        seed = random.randrange(1_000_000)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    image, metadata = shape_generator.generate_multi_shape_image(num_shapes, False)
    return seed, image, metadata


class ToyVLMGUI:
    """Tkinter GUI for the Toy VLM: scene viewer + question box + inference."""

    def __init__(self, checkpoint: str, vocab: str, scene_seed: int = None):
        if not os.path.isfile(vocab):
            sys.exit(f"Vocab file not found: '{vocab}'. Train a model first with train_model.py.")
        if not os.path.isfile(checkpoint):
            sys.exit(f"Checkpoint not found: '{checkpoint}'. Train a model first with train_model.py.")

        # Build text processor + tokenizer from the saved vocabulary.
        self.text_processor = TextProcessor()
        self.text_processor.tokenizer.load_vocab(vocab)

        # Build the model and load trained weights. A checkpoint is paired with
        # the vocabulary it was trained on: the embedding has one row per token,
        # so a vocabulary that has grown since (a new question word) makes the
        # rows not line up. Say that here rather than let load_state_dict report
        # four size mismatches and leave the reader to infer why.
        self.model = ToyVLM(self.text_processor)
        state = torch.load(checkpoint, map_location='cpu')
        ckpt_vocab = state['token_embedding.weight'].shape[0]
        vocab_size = self.text_processor.tokenizer.get_vocab_size()
        if ckpt_vocab != vocab_size:
            sys.exit(f"Checkpoint '{checkpoint}' was trained with {ckpt_vocab} "
                     f"tokens, but '{vocab}' now holds {vocab_size}. They are a "
                     f"pair: retrain with train_model.py (which rewrites the "
                     f"vocabulary) or point --checkpoint/--vocab at a matching set.")
        self.model.load_state_dict(state)

        self.device = best_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.shape_generator = ShapeGenerator()
        self.current_image = None  # (64, 64, 3) uint8 RGB
        self.current_metadata = None
        self.current_seed = None

        self.question_history = []
        self.history_index = -1

        self.root = tk.Tk()
        self.root.title("Toy Vision-Language Model")
        self.root.geometry("900x560")
        self.setup_gui()

        self.make_scene(scene_seed)

    def setup_gui(self):
        """Set up the GUI layout."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: scene image, new-scene button, seed entry.
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.canvas = tk.Canvas(
            left_frame, width=64 * CANVAS_SCALE, height=64 * CANVAS_SCALE,
            bg='black', highlightthickness=1
        )
        self.canvas.pack(pady=10)

        ttk.Button(left_frame, text="New Scene", command=self.generate_new_scene).pack(pady=(0, 10))

        seed_frame = ttk.Frame(left_frame)
        seed_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry = ttk.Entry(seed_frame, width=10)
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry.bind('<Return>', self.on_load_seed_pressed)
        ttk.Button(seed_frame, text="Load Seed", command=self.load_seed).pack(side=tk.LEFT)

        # Inner-state panel: the auxiliary count heads' readout of the current
        # image -- what the encoder believes before any reasoning happens.
        ttk.Label(left_frame, text="Encoder counts (aux heads):").pack(anchor='w')
        self.perception_display = scrolledtext.ScrolledText(
            left_frame, height=12, width=34, wrap=tk.WORD, state='disabled',
            font=('Courier', 11)
        )
        self.perception_display.pack(fill=tk.BOTH, expand=False)

        # Right panel: chat history + question entry.
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(right_frame, height=22, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Confidence bins for the model's chain: each emitted token is tinted by
        # the probability the model assigned it, so hesitation is visible at the
        # exact token where it happens (a tie's rank label, a crowded column).
        self.chat_display.tag_configure('conf_low', background='#f4a09c')     # p < .5
        self.chat_display.tag_configure('conf_mid', background='#f7c87f')     # .5-.8
        self.chat_display.tag_configure('conf_high', background='#f5eba9')    # .8-.95
        self.chat_display.tag_configure('dim', foreground='#777777')

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
        """Generate a new multi-shape RGB scene with a fresh random seed."""
        self.make_scene()

    def make_scene(self, seed=None):
        """Generate (or reproduce) a scene from a seed and update the display.

        If `seed` is None, a random one is drawn. Same seed + same code always
        gives back the same image and metadata, so a seed alone is enough to
        share and reproduce a scene.
        """
        self.current_seed, self.current_image, self.current_metadata = make_seeded_scene(
            self.shape_generator, seed
        )
        self.update_canvas_display()
        # Mirror the scene's seed into the entry, so a freshly drawn one can be
        # copied and shared without retyping it from the title bar.
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(self.current_seed))
        self.update_perception_display()
        self.root.title(f"Toy Vision-Language Model - seed: {self.current_seed}")

    def update_perception_display(self):
        """Show the auxiliary count heads' readout of the current image.

        The heads see only the vision tokens, so this is scene state, not
        question state: it updates when the scene does, and shows what the
        encoder believes before (and regardless of) any question. A count whose
        head is unsure (probability < 0.6) is marked with '?'.
        """
        image = torch.tensor(self.current_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        aux = read_aux_counts(self.model, image)
        self.perception_display.config(state='normal')
        self.perception_display.delete('1.0', tk.END)
        for family in ('shape', 'color', 'size'):
            row = "  ".join(f"{name} {count}{'' if prob >= 0.6 else '?'}"
                            for name, count, prob in aux[family])
            self.perception_display.insert(tk.END, f"{family:<6} {row}\n")
        self.perception_display.insert(
            tk.END, "\n(what the encoder reads off the image,\n before any reasoning)")
        self.perception_display.config(state='disabled')

    def load_seed(self):
        """Parse the seed entry and regenerate that exact scene."""
        text = self.seed_entry.get().strip()
        try:
            seed = int(text)
        except ValueError:
            self.add_to_chat(f"'{text}' isn't a valid seed - please enter an integer.", "System")
            return
        self.make_scene(seed)

    def on_load_seed_pressed(self, event):
        """Handle Enter key press in the seed entry."""
        self.load_seed()

    def update_canvas_display(self):
        """Update the canvas with the current RGB scene, scaled up."""
        pil_img = Image.fromarray(self.current_image, 'RGB')
        pil_img = pil_img.resize((64 * CANVAS_SCALE, 64 * CANVAS_SCALE), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

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
        rationale, answer, details = generate_response(self.model, image, question,
                                                       return_details=True)
        self.root.after(0, self.add_model_response, answer, details)

    @staticmethod
    def _confidence_tag(prob):
        """The chat tag for a token probability, or None above 0.95 (no tint)."""
        if prob < 0.5:
            return 'conf_low'
        if prob < 0.8:
            return 'conf_mid'
        if prob < 0.95:
            return 'conf_high'
        return None

    def add_model_response(self, answer, details):
        """Render the chain token by token, tinted by the model's confidence,
        then the answer and its top alternatives."""
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, "[Model] Reasoning: ")
        for word, prob in details['rationale_tokens']:
            self.chat_display.insert(tk.END, word, self._confidence_tag(prob))
            self.chat_display.insert(tk.END, " ")
        self.chat_display.insert(tk.END, "\n        Answer: ")
        for word, prob in details['answer_tokens']:
            self.chat_display.insert(tk.END, word, self._confidence_tag(prob))
            self.chat_display.insert(tk.END, " ")
        alts = "  ·  ".join(f"{w} {p:.2f}" for w, p in details['answer_topk'])
        self.chat_display.insert(tk.END, f"\n        alternatives: {alts}\n\n", 'dim')
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive GUI for the Toy VLM.")
    parser.add_argument('--checkpoint', type=str, default='toy_vlm_cot.pth')
    parser.add_argument('--vocab', type=str, default='tokenizer_vocab.json')
    parser.add_argument('--scene_seed', type=int, default=None,
                         help="Seed for the first scene shown (default: random).")
    return parser.parse_args()


def main():
    args = parse_args()
    gui = ToyVLMGUI(checkpoint=args.checkpoint, vocab=args.vocab, scene_seed=args.scene_seed)
    gui.run()


if __name__ == "__main__":
    main()
