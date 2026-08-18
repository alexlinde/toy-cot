"""
Model components for the Toy VLM.
Contains all neural network architectures and model-related functionality.

Layout (image-first, fixed positions):
  [BOS] <IMG_START> <IMG>x64 <IMG_END> <|user|> q <|assistant|>
  <THINK> rationale </THINK> <FINAL> answer </FINAL> <EOS>

The image block occupies positions 1..PREFIX_LEN-1 in every sample, so the
attention mask is prefix-LM: bidirectional within [0, PREFIX_LEN), causal after.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from text import MAX_SEQ_LEN, NUM_IMG_TOKENS, IMG_POS_START, PREFIX_LEN
from shapes import ObjType, ObjSize, COLORS, MAX_OBJECTS

# Model constants
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6

# Expensive per-forward assertions (force GPU syncs); enable with TOYCOT_DEBUG=1
DEBUG_ASSERTS = os.environ.get("TOYCOT_DEBUG", "0") == "1"

# Squeeze-and-Excitation (SE) module
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(1, c//r))
        self.fc2 = nn.Linear(max(1, c//r), c)
    def forward(self, x):  # x: (B,C,H,W)
        b,c,h,w = x.shape
        s = x.mean(dim=(2,3))                 # (B,C)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))        # (B,C)
        return x * s.view(b,c,1,1)

class VisionTokenEncoder(nn.Module):
    """Tiny CNN -> 8x8 grid tokens -> linear to hidden_dim, with ALWAYS-ON (x,y) coord channels."""

    def __init__(self, hidden_dim: int, channels: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = channels

        # Input: (B, 3, 64, 64) RGB; we concat (x,y) coords -> (B, 5, 64, 64)
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(2)  # -> (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2)  # -> (B, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(2)  # -> (B, C, 8, 8)

        # Learnable 2D positional embeddings aligned to the 8x8 conv feature map
        self.pos_embed_2d = nn.Parameter(torch.randn(1, channels, 8, 8) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        # Linear projection per-token to hidden_dim
        self.proj = nn.Linear(channels, hidden_dim)
        self.drop = nn.Dropout(p=0.1)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.se3 = SE(channels)

    @staticmethod
    def _coord_channels(B: int, H: int, W: int, device, dtype):
        """
        Returns (B, 2, H, W): x in [0,1] increasing left->right, y in [0,1] top->bottom.
        """
        y = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        # x: (B, 3, 64, 64) RGB in [0,1]
        B, C, H, W = x.shape
        if DEBUG_ASSERTS:
            assert C == 3, f"Expected RGB input with 3 channels, got {C}"
            assert H % 8 == 0 and W % 8 == 0, "Image size should be divisible by 8 for 8x8 grid downstream."

        # ALWAYS add (x,y) coordinate channels
        coords = self._coord_channels(B, H, W, x.device, x.dtype)  # (B, 2, H, W)
        x = torch.cat([x, coords], dim=1)  # (B, 5, H, W)

        # Tiny CNN stem to 8x8 feature grid
        x = F.silu(self.conv1(x)); x = self.pool1(x)
        x = F.silu(self.conv2(x)); x = self.pool2(x)
        x = F.silu(self.conv3(x)); x = self.se3(x); x = self.pool3(x)  # (B, C, 8, 8)

        # Add 2D positional encoding at the 8x8 stage
        x = x + self.pos_scale * self.pos_embed_2d

        # Flatten to tokens -> (B, NUM_IMG_TOKENS, C), raster order (row-major)
        B, C, H, W = x.shape  # 8x8
        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # Project to hidden_dim and layer-norm
        tokens = self.drop(self.proj(tokens))  # (B, N, hidden_dim)
        tokens = self.ln(tokens)

        if DEBUG_ASSERTS:
            assert tokens.size(1) == NUM_IMG_TOKENS, f"expected {NUM_IMG_TOKENS} vision tokens, got {tokens.size(1)}"
            assert torch.isfinite(tokens).all(), "[model] NaN/Inf in vision tokens"
        return tokens

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Attention capture is opt-in: set `store_attn = True` and every forward
    stashes its post-softmax weights, detached, in `last_attn` (B, heads, T, T)
    -- overwritten each call, so only the most recent forward is kept. Off by
    default, and the hot path then pays one boolean test and nothing else (no
    clone, no detach, no transfer). See generate_response(return_attention=True).
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Opt-in attention capture (plain attributes, not buffers: nothing here
        # belongs in a checkpoint).
        self.store_attn = False
        self.last_attn = None

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (T,T) bool, True = keep -> (B,h,T,T)
            m = mask[None, None, :, :]
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        if self.store_attn:
            self.last_attn = attn.detach()
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(context)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (no LayerScale)."""

    def __init__(self, d_model, num_heads, attn_dropout=0.0, resid_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.resid_drop = nn.Dropout(resid_dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn_drop(self.attn(h, h, h, mask))
        x = x + self.resid_drop(self.ffn(self.norm2(x)))
        return x


class TokenPool(nn.Module):
    """Single-query attention pooling over tokens."""
    def __init__(self, d_model: int):
        super().__init__()
        # Learned query (1,1,D); initialized small
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D] tokens
        returns: [B, D] pooled
        """
        B, N, D = x.shape
        q = self.q.expand(B, -1, -1)                     # [B,1,D]
        attn_logits = torch.matmul(q, x.transpose(1, 2)) # [B,1,N]
        attn = torch.softmax(attn_logits / math.sqrt(D), dim=-1)  # [B,1,N]
        pooled = torch.matmul(attn, x).squeeze(1)        # [B,D]
        return pooled

class AuxiliaryHeads(nn.Module):
    """Count heads over pooled vision tokens: per-shape, per-size, and per-color counts.

    Counts range 0..MAX_OBJECTS, so each head has MAX_OBJECTS + 1 classes.
    """

    def __init__(self, hidden_dim: int, num_classes: int = MAX_OBJECTS + 1):
        super().__init__()
        self.num_classes = num_classes
        self.shape_names = [obj.value for obj in ObjType]
        self.size_names = [size.value for size in ObjSize]
        self.color_names = sorted(COLORS.keys())

        self.token_pool = TokenPool(hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.shape_count_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_classes) for _ in self.shape_names
        ])
        self.size_count_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_classes) for _ in self.size_names
        ])
        self.color_count_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_classes) for _ in self.color_names
        ])

    def forward(self, vision_tokens: torch.Tensor):
        # vision_tokens: [B, N, H]
        pooled = self.token_pool(vision_tokens)
        pooled = self.pool_norm(pooled)
        pooled = self.pool_proj(pooled)

        count_logits = {
            name: head(pooled) for name, head in zip(self.shape_names, self.shape_count_heads)
        }
        size_count_logits = {
            name: head(pooled) for name, head in zip(self.size_names, self.size_count_heads)
        }
        color_count_logits = {
            name: head(pooled) for name, head in zip(self.color_names, self.color_count_heads)
        }

        return {
            'count_logits': count_logits,
            'size_count_logits': size_count_logits,
            'color_count_logits': color_count_logits,
        }


class ToyVLM(nn.Module):
    """Vision-Language Model with an image-first token prefix and prefix-LM attention."""

    def __init__(self, text_processor):
        super().__init__()

        # Vision token encoder (produces 8x8 -> 64 tokens at hidden_dim)
        self.vision_token_encoder = VisionTokenEncoder(HIDDEN_DIM)

        self.text_processor = text_processor
        vocab_size = text_processor.tokenizer.get_vocab_size()

        # Text embeddings
        self.token_embedding = nn.Embedding(vocab_size, HIDDEN_DIM)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, HIDDEN_DIM)

        # Type embeddings for text and vision tokens
        self.text_type = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.vision_type = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.input_norm = nn.LayerNorm(HIDDEN_DIM)

        # Transformer decoder (self-attention only)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(HIDDEN_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)
        ])
        self.final_norm = nn.LayerNorm(HIDDEN_DIM)
        self.dropout = nn.Dropout(0.1)

        # Prefix-LM mask: bidirectional within the fixed image prefix, causal after.
        causal = ~torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool), diagonal=1)
        prefix = torch.zeros(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool)
        prefix[:PREFIX_LEN, :PREFIX_LEN] = True
        self.register_buffer("attn_mask", causal | prefix, persistent=False)

        # Xavier initialization for linear and conv layers
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # build modules
        self.output_projection = nn.Linear(HIDDEN_DIM, vocab_size)
        self.auxiliary_heads = AuxiliaryHeads(HIDDEN_DIM)

        # init *before* tying
        self.apply(_init_weights)  # only touches Linear/Conv, not Embedding
        # GPT-style small embedding init: the default N(0,1) blows up the tied
        # output head's logits (~100 CE at init instead of ~ln(vocab))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # now tie (so Embedding keeps its intended init)
        self.output_projection.weight = self.token_embedding.weight
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def encode_image_tokens(self, images):
        """Encode image into exactly NUM_IMG_TOKENS visual tokens [B, N, H]."""
        return self.vision_token_encoder(images)

    def forward_with_embeds(self, input_embeds):
        seq_len = input_embeds.size(1)
        mask = self.attn_mask[:seq_len, :seq_len]
        hidden = input_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, mask)
        return self.output_projection(self.final_norm(hidden))

    def forward(self, images, input_tokens, return_aux=False):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        tok = self.text_processor.tokenizer
        s, n = IMG_POS_START, NUM_IMG_TOKENS

        # 1) Encode image into a fixed set of tokens
        img_tokens = self.encode_image_tokens(images)  # [B, 64, H]

        # 2) Embed input token IDs
        token_embeds = self.token_embedding(input_tokens)  # [B, T, H]

        # 3) Splice visual tokens into the fixed image-block positions
        if DEBUG_ASSERTS:
            assert seq_len >= s + n, f"sequence too short for image block: {seq_len}"
            assert (input_tokens[:, s:s + n] == tok.img_token_id).all(), \
                "[model] <IMG> placeholders not at fixed positions - layout mismatch"
        token_embeds[:, s:s + n, :] = img_tokens

        # Type embeddings: text everywhere, vision on the image block
        type_add = self.text_type.expand(batch_size, seq_len, -1).clone()
        type_add[:, s:s + n, :] = self.vision_type

        # 4) Add positional and type embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        input_embeds = token_embeds + self.position_embedding(positions) + type_add
        input_embeds = self.input_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)

        # 5) Decode with prefix-LM self-attention
        logits = self.forward_with_embeds(input_embeds)
        if DEBUG_ASSERTS:
            assert torch.isfinite(logits).all(), "[model] NaN/Inf in logits"

        aux_outputs = None
        if return_aux:
            # Provide auxiliary heads with the unmodified visual tokens
            aux_outputs = self.auxiliary_heads(img_tokens)

        return logits if not return_aux else (logits, aux_outputs)


@torch.no_grad()
def generate_response(model, image, question, max_length=160, return_rationale=True,
                      temperature=0.0, return_details=False, return_attention=False):
    """Two-stage decoding with special tokens banned from free generation.

    Prompt: [BOS] <IMG_START> <IMG>xN <IMG_END> <|user|> q <|assistant|> <THINK>
    Stage 1 generates the rationale until </THINK>; stage 2 the answer until </FINAL>.
    The answer stage is always greedy; the rationale stage samples when
    temperature > 0 (for self-consistency voting).

    Args:
        model: ToyVLM instance
        image: RGB tensor (3, 64, 64), float32 in [0,1]
        question: Question string
        max_length: Maximum steps per stage (rationale/answer)
        return_rationale: If True returns (rationale, answer); else answer
        temperature: Rationale-stage sampling temperature; 0 = greedy
        return_details: If True, returns (rationale, answer, details) where
            details exposes the inner state the decode already computed:
            'rationale_tokens' / 'answer_tokens' as [(word, prob), ...] with
            prob the untempered softmax probability of the emitted token over
            the banned-masked logits (the model's own confidence, independent
            of the sampling temperature); 'answer_topk' as the top-3
            [(word, prob), ...] alternatives at the first answer position; and
            'aux' as the auxiliary count heads' readout of the image alone,
            {'shape'|'color'|'size': [(name, count, prob), ...]} -- what the
            encoder believes before any reasoning happens.
        return_attention: If True (requires return_details), also records where
            each emitted token looked. details gains 'rationale_attn' /
            'answer_attn': lists aligned 1:1 with 'rationale_tokens' /
            'answer_tokens', each entry a CPU float32 tensor of shape
            (NUM_LAYERS, NUM_HEADS, NUM_IMG_TOKENS) holding that token's
            post-softmax attention from the query row whose logits produced it
            to the 64 image-token key positions, in the vision encoder's
            row-major raster order (index = grid_row * 8 + grid_col). Each row
            is a slice of a softmax over all keys, so it sums to the fraction of
            that head's attention spent on the image, not to 1. Costs one
            (heads, 64) slice per layer per decoding step; off, the capture
            machinery is inert.
    """
    assert return_details or not return_attention, \
        "return_attention=True is only meaningful with return_details=True"

    model.eval()
    device = next(model.parameters()).device
    tok = model.text_processor.tokenizer

    assert isinstance(image, torch.Tensor), "image must be a torch.Tensor"
    assert image.ndim == 3 and image.shape == (3, 64, 64), \
        f"expected image shape (3,64,64), got {tuple(image.shape)}"
    assert image.dtype == torch.float32, f"expected image dtype float32, got {image.dtype}"
    imin, imax = float(image.min()), float(image.max())
    assert 0.0 <= imin and imax <= 1.0, f"expected image normalized to [0,1], got range [{imin:.3f}, {imax:.3f}]"
    image = image.to(device)

    # Build prompt tokens (allow_unk for free-form questions)
    q_ids = tok.tokenize(question, allow_unk=True)
    img_block = [tok.img_start_id] + [tok.img_token_id] * NUM_IMG_TOKENS + [tok.img_end_id]
    input_ids = ([tok.bos_token_id] + img_block + [tok.user_token_id] + q_ids +
                 [tok.assistant_token_id] + [tok.think_start_id])

    all_special_ids = {
        tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id,
        tok.user_token_id, tok.assistant_token_id,
        tok.think_start_id, tok.think_end_id, tok.final_start_id, tok.final_end_id,
        tok.img_start_id, tok.img_end_id, tok.img_token_id,
    }

    # Attention capture state. `step_attn` holds the (L, H, 64) image-attention
    # of the most recent step_once call, for the caller to file against the
    # token that call emitted.
    step_attn = None

    def step_once(ids, allowed_special, temp=0.0):
        """Next token (and its untempered probability distribution) with every
        special token banned except allowed_special.

        temp=0 -> greedy argmax; temp>0 -> temperature sampling (used by
        self-consistency to diversify chains). The returned distribution is
        always the untempered softmax over the masked logits -- the model's own
        confidence, independent of how the token was drawn.
        """
        nonlocal step_attn
        ids_pad = model.text_processor.pad_sequence(ids, MAX_SEQ_LEN)
        ids_t = torch.tensor(ids_pad, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(image.unsqueeze(0), ids_t)
        pos = min(len(ids) - 1, MAX_SEQ_LEN - 1)
        if return_attention:
            # Row `pos` is the query whose logits pick this token; columns
            # IMG_POS_START..+NUM_IMG_TOKENS are the image block. Under the
            # prefix-LM mask every post-image query row can see all 64 image
            # keys in every layer, so no row is partially masked here.
            s, n = IMG_POS_START, NUM_IMG_TOKENS
            step_attn = torch.stack([
                block.attn.last_attn[0, :, pos, s:s + n]
                for block in model.transformer_blocks
            ]).float().cpu()
        next_logits = logits[0, pos, :].clone()
        banned = [tid for tid in all_special_ids if tid != allowed_special]
        next_logits[banned] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        if temp > 0:
            nxt = int(torch.multinomial(F.softmax(next_logits / temp, dim=-1), 1).item())
        else:
            nxt = int(torch.argmax(next_logits).item())
        return nxt, probs

    rationale_tokens = []
    answer_tokens = []
    rationale_probs = []
    answer_probs = []
    answer_topk = []
    rationale_attn = []
    answer_attn = []

    if return_attention:
        for block in model.transformer_blocks:
            block.attn.store_attn = True
    try:
        # Stage 1: rationale until </THINK> (sampled when temperature > 0)
        for _ in range(max_length):
            if len(input_ids) >= MAX_SEQ_LEN - 4:
                break  # leave room for </THINK> <FINAL> answer </FINAL>
            nxt, probs = step_once(input_ids, allowed_special=tok.think_end_id, temp=temperature)
            input_ids.append(nxt)
            if nxt == tok.think_end_id:
                break
            rationale_tokens.append(nxt)
            rationale_probs.append(float(probs[nxt]))
            if return_attention:
                rationale_attn.append(step_attn)
        if input_ids[-1] != tok.think_end_id:
            input_ids.append(tok.think_end_id)

        input_ids.append(tok.final_start_id)

        # Stage 2: answer until </FINAL>
        for step in range(max_length):
            if len(input_ids) >= MAX_SEQ_LEN - 1:
                break
            nxt, probs = step_once(input_ids, allowed_special=tok.final_end_id)
            if return_details and step == 0:
                # Top alternatives at the first answer position: for the one-word
                # answers most types produce, this is the whole answer's
                # distribution; for multi-word answers it is the opening word's.
                top = torch.topk(probs, k=min(4, probs.shape[0]))
                answer_topk = [(tok.decode([int(i)], skip_special_tokens=True), float(p))
                               for p, i in zip(top.values, top.indices)
                               if int(i) not in all_special_ids][:3]
            input_ids.append(nxt)
            if nxt == tok.final_end_id:
                break
            answer_tokens.append(nxt)
            answer_probs.append(float(probs[nxt]))
            if return_attention:
                answer_attn.append(step_attn)
        if input_ids[-1] != tok.final_end_id:
            input_ids.append(tok.final_end_id)
    finally:
        # Leave capture off however this exits, and drop the last (B,h,T,T)
        # buffers so they do not outlive the call.
        if return_attention:
            for block in model.transformer_blocks:
                block.attn.store_attn = False
                block.attn.last_attn = None

    rationale = tok.decode(rationale_tokens, skip_special_tokens=True)
    answer = tok.decode(answer_tokens, skip_special_tokens=True)
    if return_details:
        details = {
            'rationale_tokens': [
                (tok.decode([t], skip_special_tokens=True), p)
                for t, p in zip(rationale_tokens, rationale_probs)],
            'answer_tokens': [
                (tok.decode([t], skip_special_tokens=True), p)
                for t, p in zip(answer_tokens, answer_probs)],
            'answer_topk': answer_topk,
            'aux': read_aux_counts(model, image),
        }
        if return_attention:
            details['rationale_attn'] = rationale_attn
            details['answer_attn'] = answer_attn
        return rationale, answer, details
    return (rationale, answer) if return_rationale else answer


@torch.no_grad()
def read_aux_counts(model, image):
    """The auxiliary count heads' readout of an image: the encoder's belief
    before any reasoning happens.

    Every training forward computes these heads and inference normally discards
    them; here they are read directly off the vision tokens (no text input is
    involved). Returns {'shape'|'color'|'size': [(name, count, prob), ...]}
    where count is each head's argmax class and prob its softmax probability.

    Args:
        image: RGB tensor (3, 64, 64), float32 in [0,1].
    """
    model.eval()
    device = next(model.parameters()).device
    img_tokens = model.encode_image_tokens(image.to(device).unsqueeze(0))
    aux = model.auxiliary_heads(img_tokens)
    families = {'shape': aux['count_logits'], 'size': aux['size_count_logits'],
                'color': aux['color_count_logits']}
    out = {}
    for family, heads in families.items():
        rows = []
        for name, logits in heads.items():
            probs = F.softmax(logits[0], dim=-1)
            count = int(torch.argmax(probs).item())
            rows.append((name, count, float(probs[count])))
        out[family] = rows
    return out


@torch.no_grad()
def generate_response_self_consistent(model, image, question, k=5, temperature=0.7,
                                      max_length=160):
    """Self-consistency decoding: sample k diverse chains, majority-vote the answer.

    Rationale stages are sampled at `temperature` to diversify the chains; each
    answer stage stays greedy conditioned on its own chain. Perception errors in
    the enumerations are partly stochastic, so uncorrelated mistakes wash out in
    the vote.

    Returns (rationale, answer, chains): the winning answer, a rationale from one
    of the chains that voted for it, and all k (rationale, answer) chains.
    """
    from collections import Counter

    chains = [
        generate_response(model, image, question, max_length=max_length,
                          temperature=temperature if k > 1 else 0.0)
        for _ in range(k)
    ]

    def norm(s):
        return ' '.join(s.lower().replace('.', ' ').replace(',', ' ').split())

    votes = Counter(norm(ans) for _, ans in chains)
    winner = votes.most_common(1)[0][0]
    for rat, ans in chains:
        if norm(ans) == winner:
            return rat, ans, chains
    return chains[0][0], chains[0][1], chains
