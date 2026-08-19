"""
Interactive Tkinter GUI for the Toy VLM.

Generates random multi-shape RGB scenes, lets you ask questions about them,
and displays both the model's chain-of-thought rationale and its final
answer.

Every token of a rendered chain is clickable: clicking one paints that token's
attention over the 64 image tokens as a heat map on the scene, so the claim
'rank 3 blue cross at row 2 col 3' can be checked against where the model was
actually looking when it said it.
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
from PIL import Image, ImageDraw, ImageTk

from model import ToyVLM, generate_response
from rng import SceneRandom
from shapes import GRID_CELLS, MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator
from text import TextProcessor

CANVAS_SCALE = 4  # 64x64 scene -> 256x256 display

# Attention overlay. Magenta is the one strong hue outside the object palette
# (red/green/blue/yellow), so a hot cell never reads as the shape beneath it,
# and it stays legible on the black background.
HEAT_RGB = (255, 0, 255)
HEAT_MAX_ALPHA = 0.70
# Attention is peaky (a single cell often holds several times the next one), so
# a linear alpha ramp renders everything but the winner invisible. The gamma
# lifts the mid-range enough to see the shape of the distribution around it.
HEAT_GAMMA = 0.55

# Responses whose tokens stay clickable at once. Older ones are retired (grayed
# out, attention dropped) so a long session cannot accumulate maps forever; a
# new scene retires all of them, since their maps belong to the old image.
MAX_LIVE_RESPONSES = 8

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


def render_attention_overlay(scene_rgb, weights, scale: int = CANVAS_SCALE,
                             mark_argmax: bool = True) -> Image.Image:
    """Composite an attention heat map over a scene and return the PIL image.

    Tk canvas rectangles have no alpha, so the heat is blended into the scene
    bitmap itself and handed over as one image.

    Args:
        scene_rgb: (64, 64, 3) uint8 RGB scene.
        weights: 64 attention weights in the vision encoder's raster order
            (index = grid_row * GRID_CELLS + grid_col, row-major, matching
            VisionTokenEncoder's flatten).
        scale: display upscale factor; output is 64 * scale square.
        mark_argmax: outline the strongest cell, so the peak is unambiguous
            when several cells are close.

    Alpha is normalized per call -- the strongest cell gets HEAT_MAX_ALPHA and
    the rest scale against it through HEAT_GAMMA -- so the map shows relative
    emphasis regardless of how much of its softmax mass the token spent on the
    image at all (typically about a third; the rest goes to the text).
    """
    w = np.asarray(weights, dtype=np.float32).reshape(GRID_CELLS, GRID_CELLS)
    peak = float(w.max())
    norm = w / peak if peak > 0 else np.zeros_like(w)

    heat = np.zeros((GRID_CELLS, GRID_CELLS, 4), dtype=np.uint8)
    heat[..., 0], heat[..., 1], heat[..., 2] = HEAT_RGB
    heat[..., 3] = np.round(norm ** HEAT_GAMMA * HEAT_MAX_ALPHA * 255).astype(np.uint8)

    side = 64 * scale
    base = Image.fromarray(scene_rgb, 'RGB').resize((side, side), Image.NEAREST).convert('RGBA')
    heat_img = Image.fromarray(heat, 'RGBA').resize((side, side), Image.NEAREST)
    out = Image.alpha_composite(base, heat_img)

    if mark_argmax and peak > 0:
        cell = side // GRID_CELLS
        row, col = np.unravel_index(int(np.argmax(w)), w.shape)
        ImageDraw.Draw(out).rectangle(
            [col * cell, row * cell, (col + 1) * cell - 1, (row + 1) * cell - 1],
            outline=(255, 255, 255, 255), width=2)

    return out.convert('RGB')


def make_seeded_scene(shape_generator: ShapeGenerator, seed: int = None):
    """Generate (or reproduce) a scene from a seed alone.

    The whole derivation -- object-count draw included -- runs off one
    SceneRandom(seed) stream, the cross-language contract shared with the web
    UI (see rng.py): the same seed renders the identical scene here and in
    the browser. Global random/np.random state is left untouched.

    If `seed` is None, one is drawn first (so it can still be reported back
    to the caller). Returns (seed, image, metadata).
    """
    if seed is None:
        seed = random.randrange(1_000_000)
    rng = SceneRandom(seed)
    num_shapes = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    image, metadata = shape_generator.generate_multi_shape_image(num_shapes, False, rng=rng)
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

        # Attention-overlay state. token_attn holds only maps that belong to the
        # scene currently on the canvas: retiring a response drops its entry, so
        # a click can never paint one image's attention onto another.
        self.token_attn = {}          # unique chat tag -> (L, H, 64) CPU tensor
        self.live_responses = []      # per response, its list of clickable tags
        self.selected_tag = None
        self._token_uid = 0
        # Attention capture lives on the model itself, so two questions decoding
        # at once would read each other's maps. Questions asked while one is
        # running queue up behind this instead.
        self._infer_lock = threading.Lock()

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

        # Attention overlay controls. Heads are always averaged (4 of them, and
        # per-head maps say little at this scale); the layer choice is exposed
        # because early and late layers look at different things.
        overlay_frame = ttk.Frame(left_frame)
        overlay_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(overlay_frame, text="Attention:").pack(side=tk.LEFT, padx=(0, 4))
        self.layer_choices = ["all layers (mean)"] + [
            f"layer {i}" for i in range(len(self.model.transformer_blocks))]
        self.layer_var = tk.StringVar(value=self.layer_choices[0])
        layer_combo = ttk.Combobox(overlay_frame, textvariable=self.layer_var,
                                   values=self.layer_choices, state='readonly', width=15)
        layer_combo.pack(side=tk.LEFT, padx=(0, 4))
        layer_combo.bind('<<ComboboxSelected>>', lambda e: self.update_canvas_display())
        ttk.Button(overlay_frame, text="Clear", command=self.clear_overlay).pack(side=tk.LEFT)

        ttk.Button(left_frame, text="New Scene", command=self.generate_new_scene).pack(pady=(0, 10))

        seed_frame = ttk.Frame(left_frame)
        seed_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry = ttk.Entry(seed_frame, width=10)
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry.bind('<Return>', self.on_load_seed_pressed)
        ttk.Button(seed_frame, text="Load Seed", command=self.load_seed).pack(side=tk.LEFT)

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

        # Chain tokens are clickable while their scene is on the canvas: 'tok'
        # carries the hand-cursor affordance, 'tok_sel' marks the one being
        # shown, 'stale' grays out tokens whose scene is gone.
        self.chat_display.tag_configure('tok_sel', underline=True)
        self.chat_display.tag_configure('stale', foreground='#a0a0a0')
        self.chat_display.tag_bind('tok', '<Enter>',
                                   lambda e: self.chat_display.config(cursor='hand2'))
        self.chat_display.tag_bind('tok', '<Leave>',
                                   lambda e: self.chat_display.config(cursor=''))

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
        welcome_msg += ("\nClick any word of an answer's chain to see where the model "
                        "was looking when it emitted that word.")
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
        # A new scene invalidates every captured attention map: they were
        # measured against the image about to leave the canvas. Retire them
        # before it does, so no click can paint old attention on a new scene.
        self.retire_all_tokens()
        self.current_seed, self.current_image, self.current_metadata = make_seeded_scene(
            self.shape_generator, seed
        )
        self.update_canvas_display()
        # Mirror the scene's seed into the entry, so a freshly drawn one can be
        # copied and shared without retyping it from the title bar.
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(self.current_seed))
        self.root.title(f"Toy Vision-Language Model - seed: {self.current_seed}")

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
        """Update the canvas with the current RGB scene, scaled up, with the
        selected token's attention heat map composited over it if there is one."""
        weights = self.selected_weights()
        if weights is None:
            pil_img = Image.fromarray(self.current_image, 'RGB').resize(
                (64 * CANVAS_SCALE, 64 * CANVAS_SCALE), Image.NEAREST)
        else:
            pil_img = render_attention_overlay(self.current_image, weights)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def selected_weights(self):
        """The 64 weights to paint: the selected token's attention, averaged
        over heads and over whichever layers the selector asks for.

        None when nothing is selected (or the selection has been retired), which
        is how update_canvas_display knows to draw the bare scene.
        """
        attn = self.token_attn.get(self.selected_tag)
        if attn is None:
            return None
        choice = self.layer_var.get()
        if choice.startswith('layer '):
            return attn[int(choice.split()[1])].mean(dim=0).numpy()
        return attn.mean(dim=(0, 1)).numpy()

    def on_token_click(self, tag):
        """Show a chain token's attention. Retired tokens no-op: their map was
        measured on a scene that is no longer displayed."""
        if tag in self.token_attn:
            self.select_token(tag)

    def select_token(self, tag):
        """Mark `tag` as the shown token (None to show none) and redraw."""
        for target, add in ((self.selected_tag, False), (tag, True)):
            if target is None:
                continue
            span = self.chat_display.tag_ranges(target)
            if span:
                mark = self.chat_display.tag_add if add else self.chat_display.tag_remove
                mark('tok_sel', span[0], span[1])
        self.selected_tag = tag
        self.update_canvas_display()

    def clear_overlay(self):
        """Drop the overlay and go back to the bare scene."""
        self.select_token(None)

    def register_response(self, tags):
        """Track one response's clickable tokens, retiring the oldest responses
        once more than MAX_LIVE_RESPONSES are live."""
        if not tags:
            return  # nothing clickable (attention was not captured, or is stale)
        self.live_responses.append(tags)
        while len(self.live_responses) > MAX_LIVE_RESPONSES:
            self.retire_tokens(self.live_responses.pop(0))

    def retire_all_tokens(self):
        """Retire every live response (used when the scene changes)."""
        while self.live_responses:
            self.retire_tokens(self.live_responses.pop())

    def retire_tokens(self, tags):
        """Make these tokens unclickable and gray: drop their attention, strip
        the affordance, and delete the per-token tag (which drops its binding)."""
        if self.selected_tag in tags:
            self.select_token(None)
        for tag in tags:
            self.token_attn.pop(tag, None)
            span = self.chat_display.tag_ranges(tag)
            if span:
                self.chat_display.tag_add('stale', span[0], span[1])
                self.chat_display.tag_remove('tok', span[0], span[1])
            self.chat_display.tag_delete(tag)

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

        # The overlay belongs to a token of the previous answer; a fresh
        # question starts from the bare scene.
        self.clear_overlay()
        self.add_to_chat(question, "User")

        # Run inference in a background thread to avoid freezing the GUI.
        threading.Thread(target=self._process_question, args=(question,), daemon=True).start()

    def _process_question(self, question):
        """Run the model on the current scene and display rationale + answer."""
        image = torch.tensor(self.current_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        seed = self.current_seed  # the scene these maps will describe
        with self._infer_lock:
            rationale, answer, details = generate_response(self.model, image, question,
                                                           return_details=True,
                                                           return_attention=True)
        self.root.after(0, self.add_model_response, answer, details, seed)

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

    def _insert_token(self, word, prob, attn):
        """Insert one chain token, tinted by confidence and -- when its
        attention was captured -- clickable, under a tag unique to it.

        Returns that tag, or None for a token with no map behind it.
        """
        tags = [t for t in (self._confidence_tag(prob),) if t]
        tag = None
        if attn is not None:
            self._token_uid += 1
            tag = f"tok{self._token_uid}"
            tags += ['tok', tag]
        self.chat_display.insert(tk.END, word, tuple(tags))
        self.chat_display.insert(tk.END, " ")
        if tag is not None:
            self.token_attn[tag] = attn
            self.chat_display.tag_bind(tag, '<Button-1>',
                                       lambda e, t=tag: self.on_token_click(t))
        return tag

    def add_model_response(self, answer, details, seed=None):
        """Render the chain token by token, tinted by the model's confidence and
        each one clickable for its attention map, then the answer and its top
        alternatives.

        `seed` is the scene the answer was decoded against. If the canvas has
        moved on since (a new scene was drawn while this was decoding), the maps
        describe an image nobody can see any more, so the chain renders gray and
        unclickable rather than offering to paint them onto the wrong scene.
        """
        self.chat_display.config(state='normal')
        stale = seed is not None and seed != self.current_seed
        no_attn = [None] * (len(details['rationale_tokens']) + len(details['answer_tokens']))
        r_attn = no_attn if stale else (details.get('rationale_attn') or no_attn)
        a_attn = no_attn if stale else (details.get('answer_attn') or no_attn)
        tags = []
        start = self.chat_display.index(f"{tk.END}-1c")

        self.chat_display.insert(tk.END, "[Model] Reasoning: ")
        for (word, prob), attn in zip(details['rationale_tokens'], r_attn):
            tags.append(self._insert_token(word, prob, attn))
        self.chat_display.insert(tk.END, "\n        Answer: ")
        for (word, prob), attn in zip(details['answer_tokens'], a_attn):
            tags.append(self._insert_token(word, prob, attn))
        alts = "  ·  ".join(f"{w} {p:.2f}" for w, p in details['answer_topk'])
        self.chat_display.insert(tk.END, f"\n        alternatives: {alts}\n\n", 'dim')
        if stale:
            self.chat_display.tag_add('stale', start, tk.END)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

        self.register_response([t for t in tags if t is not None])

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
