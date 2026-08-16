"""
Text processing module for the Toy VLM.
Updated to support prefixing image tokens in the sequence and
multi-turn-friendly special tokens for user/assistant and reasoning spans.

Rationales may be passed to TextProcessor.prepare_input_sequence either as a
plain string (fully supervised, the historical behavior) or as a list of
``(text, supervised)`` segments. Unsupervised segments still contribute their
tokens to ``input_ids`` -- the model reads them as context -- but their target
positions are zeroed in ``rat_mask``, which is what makes error-injected
self-correction traces trainable: the model learns to recover from a wrong
enumeration fact without ever being taught to emit one
(see questions.inject_correction).
"""

import re
import json
import os
from typing import List, Sequence, Set, Tuple, Union

try:  # torch is only needed to decode tensors; the data layer must run without it
    import torch
except ImportError:  # pragma: no cover - exercised by torch-free tooling
    torch = None

# Token/sequence constants
# 256 leaves room for the dense-scene traces: a 12-object enumeration is ~96
# tokens and a depth-4 chain enumerates five descriptor sets. The fixed
# overhead is 74 tokens (see questions.WORD_BUDGET).
MAX_SEQ_LEN = 256
NUM_IMG_TOKENS = 64

# Image-first layout:
#   position 0            : <BOS>
#   position 1            : <IMG_START>
#   positions 2..2+63     : <IMG> placeholders
#   position 66           : <IMG_END>
#   position 67 onwards   : <|user|> question ...
IMG_POS_START = 2
PREFIX_LEN = 2 + NUM_IMG_TOKENS + 1  # 67: BOS + IMG_START + 64 <IMG> + IMG_END

# A rationale is either one fully supervised string or a list of segments.
RationaleSegment = Tuple[str, bool]
RationaleSpec = Union[str, Sequence[RationaleSegment], None]


def as_segments(rationale: RationaleSpec) -> List[RationaleSegment]:
    """Normalize a rationale into a list of ``(text, supervised)`` segments.

    A plain string is one fully supervised segment, so every existing call site
    keeps its exact behavior. Segment texts are concatenated *verbatim* (no
    separator is inserted), so the caller owns the ' . ' step separators and the
    concatenation is the rationale the model actually reads.
    """
    if rationale is None:
        return []
    if isinstance(rationale, str):
        return [(rationale, True)]
    segments: List[RationaleSegment] = []
    for segment in rationale:
        text, supervised = segment
        if not isinstance(text, str):
            raise TypeError(f"[text] rationale segment text must be str, got {text!r}")
        segments.append((text, bool(supervised)))
    return segments

class SimpleTokenizer:
    """Simple word-based tokenizer for the toy VLM (word-level)."""
    
    # The 13 fixed special tokens always occupy ids 0..12.
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
        '<|user|>': 4,
        '<|assistant|>': 5,
        '<THINK>': 6,
        '</THINK>': 7,
        '<FINAL>': 8,
        '</FINAL>': 9,
        '<IMG_START>': 10,
        '<IMG_END>': 11,
        '<IMG>': 12,
    }

    def __init__(self, vocab_file: str = None):
        if vocab_file and os.path.exists(vocab_file):
            # Load pretrained vocabulary
            self.load_vocab(vocab_file)
        else:
            # Initialize with base vocabulary for dialog + image-token format
            self._reset_special_tokens()

    def _reset_special_tokens(self):
        """Reset the vocabulary to just the fixed special tokens."""
        self.vocab = dict(self.SPECIAL_TOKENS)
        self._update_mappings()

    def _update_mappings(self):
        """Update reverse mapping and special token IDs."""
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        # Special token IDs
        self.pad_token_id = self.vocab['<PAD>']
        self.bos_token_id = self.vocab['<BOS>']
        self.eos_token_id = self.vocab['<EOS>']
        self.unk_token_id = self.vocab['<UNK>']

        # Conversation and reasoning markers
        self.user_token_id = self.vocab['<|user|>']
        self.assistant_token_id = self.vocab['<|assistant|>']
        self.think_start_id = self.vocab['<THINK>']
        self.think_end_id = self.vocab['</THINK>']
        self.final_start_id = self.vocab['<FINAL>']
        self.final_end_id = self.vocab['</FINAL>']

        # Image markers
        self.img_start_id = self.vocab['<IMG_START>']
        self.img_end_id = self.vocab['<IMG_END>']
        self.img_token_id = self.vocab['<IMG>']
    
    def build_vocab(self, words: Set[str]):
        """Build a deterministic vocabulary from an explicit word set.

        Ids are assigned to sorted(words) after the fixed special tokens, so
        the mapping depends only on the word set -- never on sampling luck.
        """
        self._reset_special_tokens()

        next_idx = len(self.vocab)
        for word in sorted(words):
            if word not in self.vocab:
                self.vocab[word] = next_idx
                next_idx += 1

        self._update_mappings()
        return self.vocab

    def build_vocab_from_rationales(self, rationale_generator, num_samples=200):
        """Build the vocabulary from a RationaleGenerator's declared word set.

        `num_samples` is ignored: the vocabulary is enumerated deterministically
        by the generator rather than sampled. The argument is kept so existing
        call sites keep working.
        """
        self.build_vocab(rationale_generator.vocabulary())
        print(f"Built vocabulary with {len(self.vocab)} tokens from RationaleGenerator")

    def _extract_words(self, text: str) -> Set[str]:
        """Extract normalized words from text."""
        if not text:
            return set()
        
        normalized = self._preprocess_text(text)
        return set(normalized.split()) if normalized else set()
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2, sort_keys=True)
        print(f"Saved vocabulary to {vocab_file}")
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self._update_mappings()
        print(f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}")
    
    @classmethod
    def load_pretrained(cls, vocab_file: str):
        """Create tokenizer instance from saved vocabulary."""
        tokenizer = cls()
        tokenizer.load_vocab(vocab_file)
        return tokenizer
    
    def _preprocess_text(self, text: str) -> str:
        """Lowercase, keep '.' as a standalone token, drop everything else."""
        text = text.lower()
        # '.' is a real token (the rationale step separator)
        text = text.replace('.', ' . ')
        # Keep only lowercase letters, digits, dots and spaces
        text = re.sub(r'[^a-z0-9. ]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
        
    def tokenize(self, text: str, allow_unk: bool = False) -> List[int]:
        """Convert text to token IDs.

        Args:
            text: Text to tokenize
            allow_unk: If False, raise error on unknown words (for training).
                      If True, use <UNK> token (for inference).
        """
        text = self._preprocess_text(text)
        words = text.split()

        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                if not allow_unk:
                    raise ValueError(f"Unknown word '{word}' in text '{text}'. "
                                   f"Vocabulary may not be properly built. "
                                   f"Available words: {sorted(self.vocab.keys())[:20]}...")
                tokens.append(self.unk_token_id)

        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        if torch is not None and isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        words = []
        for token_id in tokens:
            if token_id in self.idx_to_word:
                word = self.idx_to_word[token_id]
                if skip_special_tokens and word.startswith('<') and word.endswith('>'):
                    continue
                words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

class TextProcessor:
    """Handles text processing tasks for the VLM."""
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
    
    def prepare_input_sequence(
        self,
        question: str,
        answer: str = None,
        rationale: RationaleSpec = None
    ) -> tuple:
        """
        Compose sequence per image-first format:

        input_ids  = [BOS] + img_block + [USER] + q_ids + [ASSIST]
                     + [THINK] + r_ids + [THINK_END]
                     + [FINAL] + a_ids + [FINAL_END] + [EOS]
        target_ids = input_ids[1:]

        The image block occupies a fixed span (positions 1..PREFIX_LEN-1), so
        the model can build its attention mask from PREFIX_LEN/IMG_POS_START.

        `rationale` is either a plain string (every rationale token supervised)
        or a list of ``(text, supervised)`` segments. Tokens of an unsupervised
        segment are part of input_ids like any other -- the model conditions on
        them -- but the target positions that *predict* them carry rat_mask=0.
        """
        tok = self.tokenizer

        q_tokens = tok.tokenize(question)
        a_tokens = tok.tokenize(answer) if answer is not None else []

        # Rationale tokens, plus a per-token supervision flag from the segment
        # each token came from. Tokenizing segment by segment is equivalent to
        # tokenizing their concatenation as long as segment boundaries fall on
        # whitespace, which the ' . ' step separator guarantees.
        r_tokens: List[int] = []
        r_supervised: List[int] = []
        for text, supervised in as_segments(rationale):
            segment_tokens = tok.tokenize(text)
            r_tokens.extend(segment_tokens)
            r_supervised.extend([1 if supervised else 0] * len(segment_tokens))

        img_block = [tok.img_start_id] + [tok.img_token_id] * NUM_IMG_TOKENS + [tok.img_end_id]

        input_ids = (
            [tok.bos_token_id] +
            img_block +
            [tok.user_token_id] + q_tokens +
            [tok.assistant_token_id] +
            [tok.think_start_id] + r_tokens + [tok.think_end_id] +
            [tok.final_start_id] + a_tokens + [tok.final_end_id] +
            [tok.eos_token_id]
        )

        # A sample that overflows the context must fail loudly: silently
        # truncating a sequence would cut the CoT supervision to pieces.
        assert len(input_ids) <= MAX_SEQ_LEN, (
            f"[text] Sequence of {len(input_ids)} tokens exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            f"question={question!r} rationale={rationale!r} answer={answer!r}"
        )

        # Standard next-token targets: predict input_ids shifted by 1
        target_ids = input_ids[1:]

        # Build span masks per policy (masks align to target_ids = input_ids[1:]):
        # We supervise predictions of tokens strictly inside each span.
        # - rat_mask: supervise predictions of rationale content tokens only
        # - ans_mask: supervise predictions of answer content tokens only
        rat_mask = [0] * len(target_ids)
        ans_mask = [0] * len(target_ids)

        # Supervision windows defined on source positions (k indexes input_ids[k] -> predicts input_ids[k+1])
        def find_pos(tid):
            try:
                return input_ids.index(tid)
            except ValueError:
                return -1

        th_s = find_pos(tok.think_start_id)
        th_e = find_pos(tok.think_end_id)
        fn_s = find_pos(tok.final_start_id)
        fn_e = find_pos(tok.final_end_id)

        # Map to target positions: target[k] == input_ids[k+1]
        # Supervise content tokens AND the closing tag for each span. Inside the
        # THINK span a content token is supervised only if its segment was:
        # offset counts rationale tokens, and offset == len(r_tokens) is the
        # </THINK> tag itself, which is always supervised.
        for k in range(len(target_ids)):
            pred_idx = k + 1  # index into input_ids of the token being predicted
            if th_s != -1 and th_e != -1 and th_s < pred_idx <= th_e:
                offset = pred_idx - th_s - 1
                rat_mask[k] = 1 if offset >= len(r_tokens) else r_supervised[offset]
            if fn_s != -1 and fn_e != -1 and fn_s < pred_idx <= fn_e:
                ans_mask[k] = 1

        # Diagnostics and structural assertions
        # Ensure marker order and presence
        def find_pos(tid):
            try:
                return input_ids.index(tid)
            except ValueError:
                return -1

        bos_p = 0 if len(input_ids) > 0 and input_ids[0] == tok.bos_token_id else -1
        usr_p = find_pos(tok.user_token_id)
        asst_p = find_pos(tok.assistant_token_id)
        th_s = find_pos(tok.think_start_id)
        th_e = find_pos(tok.think_end_id)
        fn_s = find_pos(tok.final_start_id)
        fn_e = find_pos(tok.final_end_id)
        eos_p = len(input_ids)-1 if len(input_ids) and input_ids[-1] == tok.eos_token_id else -1

        assert bos_p == 0, "[text] <BOS> must be at position 0"
        img_s = find_pos(tok.img_start_id)
        img_e = find_pos(tok.img_end_id)
        assert img_s == 1, "[text] <IMG_START> must be at position 1"
        assert img_e == PREFIX_LEN - 1, (
            f"[text] <IMG_END> must be at position {PREFIX_LEN - 1}, found {img_e}"
        )
        assert all(t == tok.img_token_id for t in input_ids[IMG_POS_START:img_e]), (
            f"[text] Image placeholders must fill positions {IMG_POS_START}..{img_e - 1}"
        )
        # Check placeholders count
        num_placeholders = sum(1 for t in input_ids if t == tok.img_token_id)
        assert num_placeholders == NUM_IMG_TOKENS, f"[text] Expected {NUM_IMG_TOKENS} <IMG> tokens, found {num_placeholders}"
        assert usr_p == PREFIX_LEN, (
            f"[text] <|user|> must directly follow the image block at {PREFIX_LEN}, found {usr_p}"
        )
        assert asst_p > usr_p, "[text] <|assistant|> must come after <|user|>"
        # THINK and FINAL sections exist even if empty strings were passed (they may be contiguous)
        assert th_s > asst_p and th_e > th_s, "[text] Invalid THINK span"
        assert th_e - th_s - 1 == len(r_tokens), (
            f"[text] THINK span holds {th_e - th_s - 1} tokens, "
            f"segments produced {len(r_tokens)}"
        )
        assert fn_s > th_e and fn_e > fn_s, "[text] Invalid FINAL span"
        assert eos_p == len(input_ids)-1, "[text] <EOS> must be final token"

        # Segment-level supervision must land exactly on the rationale span:
        # one supervised target per supervised rationale token, plus </THINK>.
        assert sum(rat_mask) == sum(r_supervised) + 1, (
            f"[text] rat_mask supervises {sum(rat_mask)} positions, expected "
            f"{sum(r_supervised) + 1} (supervised rationale tokens + </THINK>)"
        )
        assert sum(ans_mask) == len(a_tokens) + 1, (
            f"[text] ans_mask supervises {sum(ans_mask)} positions, expected "
            f"{len(a_tokens) + 1} (answer tokens + </FINAL>)"
        )

        # Mask policy sanity: special markers must be unsupervised
        special_set = {
            tok.bos_token_id, tok.user_token_id, tok.assistant_token_id,
            tok.think_start_id, tok.think_end_id, tok.final_start_id, tok.final_end_id,
            tok.eos_token_id, tok.img_start_id, tok.img_end_id, tok.img_token_id
        }
        allowed_special_to_supervise = {tok.think_end_id, tok.final_end_id}
        for k, tid in enumerate(target_ids):
            if tid in special_set and tid not in allowed_special_to_supervise:
                assert rat_mask[k] == 0 and ans_mask[k] == 0, f"[text] Mask error: supervising special token id={tid} at k={k}"

        return input_ids, target_ids, rat_mask, ans_mask

    def pad_sequence(self, tokens, max_length=MAX_SEQ_LEN):
        pad_id = self.tokenizer.pad_token_id
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [pad_id] * (max_length - len(tokens))
            
    def clean_response(self, response: str) -> str:
        """Clean up generated response text."""
        # Remove extra spaces and normalize
        response = ' '.join(response.split())
        
        # Ensure proper punctuation spacing
        response = response.replace(' ?', '?').replace(' .', '.')
        
        return response.strip()