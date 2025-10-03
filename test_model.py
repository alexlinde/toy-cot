"""
Interactive execution module for the Toy VLM.
Loads a trained model and allows interactive Q&A with generated shapes.
"""

import torch
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import threading

from shapes import ShapeGenerator
from text import SimpleTokenizer, TextProcessor
from model import ToyVLM, generate_response
from shapes import ObjType, ObjSize
import random
from runtime import setup_runtime

def get_model_stats(model):
    """Get comprehensive model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Component-wise parameter counts
    vision_params = sum(p.numel() for p in model.vision_token_encoder.parameters())
    text_embed_params = sum(p.numel() for p in model.token_embedding.parameters()) + \
                      sum(p.numel() for p in model.position_embedding.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_blocks.parameters())
    output_params = sum(p.numel() for p in model.output_projection.parameters())

    # Model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'vision_params': vision_params,
        'text_embed_params': text_embed_params,
        'transformer_params': transformer_params,
        'output_params': output_params,
        'vocab_size': model.output_projection.out_features,
        'hidden_dim': model.token_embedding.embedding_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': getattr(model.transformer_blocks[0].attn, 'num_heads', None),
        'device': str(next(model.parameters()).device),
        'model_size_mb': model_size_mb
    }

def format_number(num):
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

class ToyVLMGUI:
    """Tkinter GUI for the Toy VLM."""
    
    def __init__(self, model_path='toy_vlm_cot.pth', tokenizer_vocab='tokenizer_vocab.json'):
        # Initialize text processing with pretrained tokenizer
        self.tokenizer = SimpleTokenizer.load_pretrained(tokenizer_vocab)
        self.text_processor = TextProcessor()
        self.text_processor.tokenizer = self.tokenizer

        # Setup runtime (device/AMP)
        self.runtime = setup_runtime()

        # Initialize model components (with CoT support: 6 layers)
        self.model = ToyVLM(self.text_processor)

        # Cross-device safe load (e.g., trained on CUDA/bf16, run on MPS/CPU)
        try:
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
        except FileNotFoundError:
            print(f"Warning: Model weights not found at '{model_path}'. Using untrained model.")
        except Exception as e:
            print(f"Warning: Failed to load weights from '{model_path}': {e}. Using untrained model.")

        # Move to target device and eval
        self.model = self.runtime["to_device"](self.model)
        self.model.eval()

        self.shape_generator = ShapeGenerator()

        self.current_shape_type = None
        self.current_image = None
        self.current_metadata = None  # Store metadata for multi-shape images

        # CoT display toggle
        self.show_rationale = True
        
        # Question history for navigation
        self.question_history = []
        self.history_index = -1
        
        # Image editing state
        self.editing_mode = ObjType.SQUARE.value  # Default to square
        self.erase_mode = False
        self.tool_size = 10
        self.canvas_scale = 300  # Scale factor from 64x64 to display size
        self.is_drawing = False
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Toy Vision-Language Model")
        self.root.geometry("800x500")
        self.setup_gui()
        
        # Generate initial shape
        self.generate_new_shape()
    
    def setup_gui(self):
        """Set up the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for image
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Image display (using Canvas for editing)
        self.canvas = tk.Canvas(left_frame, width=self.canvas_scale, height=self.canvas_scale, bg='black', highlightthickness=1)
        self.canvas.pack(pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # # Tool buttons
        edit_frame = ttk.Frame(left_frame)
        edit_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Shape selection radio buttons
        shapes_frame = ttk.Frame(edit_frame)        
        shapes_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.tool_var = tk.StringVar(value=ObjType.SQUARE.value)
        ttk.Radiobutton(shapes_frame, text="Square", variable=self.tool_var, 
                       value=ObjType.SQUARE.value, command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(shapes_frame, text="Circle", variable=self.tool_var, 
                       value=ObjType.CIRCLE.value, command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        self.erase_var = tk.BooleanVar()
        self.noise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(shapes_frame, text="Erase Mode", variable=self.erase_var, 
                       command=self.on_erase_change).pack(side=tk.RIGHT, padx=5)
        # Noise checkbox
        ttk.Checkbutton(shapes_frame, text="Add Noise", variable=self.noise_var).pack(side=tk.RIGHT, padx=5)
        
        # Size slider
        size_frame = ttk.Frame(edit_frame)
        size_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.size_var = tk.IntVar(value=10)
        self.size_slider = ttk.Scale(size_frame, from_=5, to=30, orient=tk.HORIZONTAL, 
                                    variable=self.size_var, command=self.on_size_change)
        self.size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.size_label = ttk.Label(size_frame, text="10")
        self.size_label.pack(side=tk.LEFT)
        
        # Generate new shape button
        ttk.Button(edit_frame, text="New Shape", command=self.generate_new_shape).pack(pady=5, side=tk.LEFT)

                
        # Right panel for chat
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat history
        self.chat_display = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Question input
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(input_frame, text="Ask a question:").pack(anchor='w')

        # Entry and button in the same row
        entry_button_frame = ttk.Frame(input_frame)
        entry_button_frame.pack(fill=tk.X, pady=(5, 10))

        self.question_entry = ttk.Entry(entry_button_frame)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.question_entry.bind('<Return>', self.on_enter_pressed)
        self.question_entry.bind('<Up>', self.on_up_key)
        self.question_entry.bind('<Down>', self.on_down_key)

        # Send button
        ttk.Button(entry_button_frame, text="Ask Question", command=self.ask_question).pack(side=tk.RIGHT)

        # Add CoT toggle checkbox
        cot_frame = ttk.Frame(input_frame)
        cot_frame.pack(fill=tk.X, pady=(0, 5))
        self.show_rationale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cot_frame, text="Show Chain-of-Thought Reasoning",
                       variable=self.show_rationale_var,
                       command=self.on_rationale_toggle).pack(anchor='w')

        # Display model statistics
        self.display_model_stats()

        # Add initial welcome message
        welcome_msg = "🧠 Chain-of-Thought VLM Ready!\n"
        welcome_msg += "Ask questions about the shapes:\n"
        welcome_msg += "• 'what shapes do you see?'\n"
        welcome_msg += "• 'how many circles are there?'\n"
        welcome_msg += "• 'are there more circles than squares?'\n"
        welcome_msg += "• 'how many red shapes are there?'"
        self.add_to_chat(welcome_msg, "System")

        # Give focus to question entry
        self.question_entry.focus_set()
    
    def display_model_stats(self):
        """Display model statistics in the chat."""
        stats = get_model_stats(self.model)

        stats_text = f"📊 Model Statistics:\n"
        stats_text += f"• Total Parameters: {format_number(stats['total_params'])}\n"
        stats_text += f"• Vision Encoder: {format_number(stats['vision_params'])}\n"
        stats_text += f"• Text Embeddings: {format_number(stats['text_embed_params'])}\n"
        stats_text += f"• Transformer Blocks: {format_number(stats['transformer_params'])}\n"
        stats_text += f"• Output Layer: {format_number(stats['output_params'])}\n"
        stats_text += f"• Model Size: {stats['model_size_mb']:.1f} MB\n"
        stats_text += f"• Architecture: {stats['hidden_dim']}d, {stats['num_layers']} layers, {stats['num_heads']} heads\n"
        stats_text += f"• Vocabulary Size: {format_number(stats['vocab_size'])}\n"
        stats_text += f"• Device: {stats['device']}"

        self.add_to_chat(stats_text, "System")

    def add_to_chat(self, message, sender="User"):
        """Add a message to the chat display."""
        self.chat_display.config(state='normal')
        if sender == "System":
            self.chat_display.insert(tk.END, f"🤖 {message}\n\n")
        elif sender == "User":
            self.chat_display.insert(tk.END, f"👤 {message}\n")
        else:  # VLM response
            self.chat_display.insert(tk.END, f"🎯 {message}\n\n")

        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def generate_new_shape(self):
        """Generate a new multi-shape RGB image and update the display."""
        # Generate multi-shape RGB image with metadata
        num_shapes = random.randint(1, 8)
        self.current_image, self.current_metadata = self.shape_generator.generate_multi_shape_image(
            num_shapes, add_noise=bool(self.noise_var.get())
        )
        self.update_canvas_display()

        # Display shape info in chat
        shape_info = f"Generated {len(self.current_metadata)} shapes: "
        shape_summary = [f"{m['size_category']} {m['shape']}" for m in self.current_metadata]
        shape_info += ", ".join(shape_summary)
        # self.add_to_chat(shape_info, "System")
    
    def update_canvas_display(self):
        """Update the canvas with the current image."""
        # Convert numpy array to PIL Image and then to PhotoImage
        # Handle both RGB (H, W, 3) and grayscale (H, W) images
        img_array = self.current_image
        if img_array.dtype != np.uint8:
            img_uint8 = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img_array
        pil_img = Image.fromarray(img_uint8, mode='L')

        self.img_size = pil_img.size
        pil_img = pil_img.resize((self.canvas_scale, self.canvas_scale), Image.NEAREST)  # Scale up with nearest neighbor

        self.photo = ImageTk.PhotoImage(pil_img)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.photo, anchor='center')
        
    def on_enter_pressed(self, event):
        """Handle Enter key press in question entry."""
        self.ask_question()
    
    def on_up_key(self, event):
        """Handle Up arrow key press - navigate to previous question in history."""
        if not self.question_history:
            return
        
        # If at end of history, move to last item
        if self.history_index == -1:
            self.history_index = len(self.question_history) - 1
        # Otherwise move backwards
        elif self.history_index > 0:
            self.history_index -= 1
        
        # Load the question at current index
        if 0 <= self.history_index < len(self.question_history):
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])
    
    def on_down_key(self, event):
        """Handle Down arrow key press - navigate to next question in history."""
        if not self.question_history or self.history_index == -1:
            return
        
        # Move forward in history
        if self.history_index < len(self.question_history) - 1:
            self.history_index += 1
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])
        else:
            # At end of history, clear entry and reset index
            self.history_index = -1
            self.question_entry.delete(0, tk.END)
    
    def ask_question(self):
        """Process a question about the current shape."""
        question = self.question_entry.get().strip()
        if not question:
            return
        
        # Clear the input
        self.question_entry.delete(0, tk.END)
        
        # Add question to history and reset history index
        self.question_history.append(question)
        self.history_index = -1
        
        # Add question to chat
        self.add_to_chat(question, "User")
        
        # Process in background thread to avoid freezing GUI
        threading.Thread(target=self._process_question, args=(question,), daemon=True).start()
    
    def _process_question(self, question):
        """Process the question in a background thread."""
        # Convert numpy array to torch tensor in (C,H,W)
        image_tensor = torch.from_numpy(self.current_image).unsqueeze(0).float() / 255.0

        # Generate response with chain-of-thought
        with self.runtime["autocast"]:
            rationale, answer = generate_response(
                self.model, image_tensor, question, max_length=35, return_rationale=True
            )

        # Format response based on show_rationale setting
        if self.show_rationale_var.get() and rationale:
            response = f"💭 Reasoning: {rationale}\n\n✓ Answer: {answer}"
        else:
            response = answer

        # Update GUI in main thread
        self.root.after(0, self.add_to_chat, response, "VLM")

    def on_rationale_toggle(self):
        """Handle rationale display toggle."""
        self.show_rationale = self.show_rationale_var.get()
    
    def on_tool_change(self):
        """Handle tool selection change."""
        self.editing_mode = self.tool_var.get()
    
    def on_erase_change(self):
        """Handle erase mode checkbox change."""
        self.erase_mode = self.erase_var.get()
    
    def on_size_change(self, value):
        """Handle size slider change."""
        self.tool_size = int(float(value))
        self.size_label.config(text=str(self.tool_size))
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas."""
        self.is_drawing = True
        self.draw_at_position(event.x, event.y)
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas."""
        if self.is_drawing:
            self.draw_at_position(event.x, event.y)
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas."""
        _ = event  # Unused parameter
        self.is_drawing = False
    
    def draw_at_position(self, canvas_x, canvas_y):
        """Draw at the specified canvas position."""
        # Convert canvas coordinates to image coordinates (300x300 -> 64x64)
        img_x = int(canvas_x * self.img_size[0] / self.canvas_scale)
        img_y = int(canvas_y * self.img_size[1] / self.canvas_scale)
        
        # Ensure coordinates are within bounds
        if 0 <= img_x < self.img_size[0] and 0 <= img_y < self.img_size[1]:
            self.draw_shape(self.editing_mode, img_x, img_y, self.tool_size, 0 if self.erase_mode else 255)
            self.update_canvas_display()
    
    def draw_shape(self, shape_type, center_x, center_y, size, fill_color):
        """Draw or erase a shape at the specified position using Pillow."""
        # Convert numpy array to PIL Image
        img_array = (self.current_image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array, mode='L')
        draw = ImageDraw.Draw(pil_img)
        half_size = size // 2
        x1 = center_x - half_size; y1 = center_y - half_size;
        x2 = center_x + half_size; y2 = center_y + half_size;

        if shape_type == ObjType.SQUARE.value:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        elif shape_type == ObjType.CIRCLE.value:
            draw.ellipse([x1, y1, x2, y2], fill=fill_color)

        # Convert back to numpy array
        self.current_image = np.array(pil_img, dtype=np.float32) / 255.0
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()

def main():    
    try:
        gui = ToyVLMGUI()
        gui.run()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    main()