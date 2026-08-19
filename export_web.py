"""
Export the Toy VLM for the browser (web/): ONNX graphs, vocab, manifest,
and golden fixtures the TypeScript port is tested against.

Outputs (under --out, default web/):
  public/model/step.onnx   one decode step:
                             (image (1,3,64,64) f32, ids (1,256) i64, pos (1,) i64)
                             -> logits_row (V,), attn_row (L, H, 64)
                           logits_row is the next-token logits at `pos`;
                           attn_row is that query row's post-softmax attention
                           to the 64 image-token keys, per layer and head --
                           the same slice generate_response(return_attention)
                           captures via store_attn.
  public/model/aux.onnx    image -> (shape (4,C), size (3,C), color (4,C))
                           count-head logits, rows in manifest name order.
  public/model/vocab.json  copy of the tokenizer vocabulary.
  public/model/manifest.json
                           architecture stats, sequence constants, special
                           token ids, aux head name orders, file hashes.
  fixtures/rng.json        PCG32 + derived-draw golden vectors.
  fixtures/scenes.json     seed -> metadata + raw RGB bytes (base64) + sha256.
  fixtures/transcripts.json
                           (seed, question) -> per-token words/probs, answer
                           top-k, aux readout, attention summaries. Produced
                           on CPU with the real generate_response, so the TS
                           decode loop is verified end-to-end against the
                           exact code path the Python GUI uses.

The export refuses to write a bundle whose ONNX outputs disagree with the
torch model beyond fp32 noise: parity asserts run first.
"""

import argparse
import base64
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import ToyVLM, generate_response, read_aux_counts
from rng import SceneRandom, PCG_INITSEQ
from shapes import ShapeGenerator, MIN_OBJECTS, MAX_OBJECTS, GRID_CELLS
from text import TextProcessor, MAX_SEQ_LEN, NUM_IMG_TOKENS, IMG_POS_START, PREFIX_LEN

FIXTURE_RNG_SEEDS = [0, 1, 7, 42, 4711, 123456789, 999999]
FIXTURE_SCENE_SEEDS = [7, 42, 777, 4711, 12345, 31337, 99999, 123456, 555555, 999999,
                       2, 13, 64, 2024, 424242, 68301, 8, 91, 700001, 313]
# One question per question style the GUI advertises, asked across several
# scenes so the transcripts cover short and long chains.
FIXTURE_QA = [
    (7, "is there a red circle"),
    (7, "how many circles are there"),
    (42, "is there a square above a circle"),
    (42, "which shape is second from the left"),
    (777, "is the number of squares greater than the number of circles"),
    (4711, "are there any triangles on the left"),
    (12345, "what shape is fourth from the right"),
    (12345, "how many red shapes are there"),
    (31337, "is there a blue cross"),
    (99999, "which shape is closest to the top"),
]


class StepModel(torch.nn.Module):
    """One decode step as a pure function, for ONNX export.

    Calls the real ToyVLM.forward (so there is no duplicated math to drift)
    with store_attn enabled; tracing captures the per-layer attention tensors
    the store_attn hook records, and the graph returns the row `pos` slices
    that generate_response() reads.
    """

    def __init__(self, vlm: ToyVLM):
        super().__init__()
        self.vlm = vlm
        for block in vlm.transformer_blocks:
            block.attn.store_attn = True

    def forward(self, image, ids, pos):
        logits = self.vlm(image, ids)                      # (1, T, V)
        s, n = IMG_POS_START, NUM_IMG_TOKENS
        attn = torch.stack([
            block.attn.last_attn[:, :, :, s:s + n]         # (1, H, T, 64)
            for block in self.vlm.transformer_blocks
        ], dim=1)                                          # (1, L, H, T, 64)
        logits_row = logits.index_select(1, pos).reshape(-1)          # (V,)
        attn_row = attn.index_select(3, pos)               # (1, L, H, 1, 64)
        attn_row = attn_row.reshape(attn.shape[1], attn.shape[2], n)  # (L, H, 64)
        return logits_row, attn_row


class AuxModel(torch.nn.Module):
    """Aux count-head logits per family, rows in the manifest's name order."""

    def __init__(self, vlm: ToyVLM):
        super().__init__()
        self.vlm = vlm

    def forward(self, image):
        tokens = self.vlm.encode_image_tokens(image)
        aux = self.vlm.auxiliary_heads(tokens)
        heads = self.vlm.auxiliary_heads
        shape = torch.stack([aux['count_logits'][k][0] for k in heads.shape_names])
        size = torch.stack([aux['size_count_logits'][k][0] for k in heads.size_names])
        color = torch.stack([aux['color_count_logits'][k][0] for k in heads.color_names])
        return shape, size, color


def sha256_file(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def image_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    """(64,64,3) uint8 -> (3,64,64) float32 in [0,1], exactly as the GUI does."""
    return torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0


def model_stats(model: ToyVLM) -> dict:
    """Parameter counts for the manifest (mirrors test_model.get_model_stats,
    which is not imported because that module pulls in tkinter)."""
    count = lambda m: sum(p.numel() for p in m.parameters())
    return {
        'total_params': count(model),
        'vision_params': count(model.vision_token_encoder),
        'aux_params': count(model.auxiliary_heads),
        'transformer_params': count(model.transformer_blocks),
        'hidden_dim': model.token_embedding.embedding_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': model.transformer_blocks[0].attn.num_heads,
        'vocab_size': model.output_projection.out_features,
    }


def export_onnx(model: ToyVLM, model_dir: str) -> None:
    step = StepModel(model).eval()
    aux = AuxModel(model).eval()

    image = torch.zeros(1, 3, 64, 64)
    ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    pos = torch.tensor([67], dtype=torch.long)

    # dynamo=True: the deprecated tracing exporter emits float64 copies of
    # the vision-encoder weights on this torch/numpy combination, which
    # onnxruntime rejects at load. The dynamo path exports cleanly, including
    # the store_attn attribute capture (verified: the attention output flows
    # into the graph correctly).
    # external_data=False: weights embedded in the .onnx file itself -- the
    # browser then fetches one file per graph and the manifest hash covers
    # the whole model.
    torch.onnx.export(
        step, (image, ids, pos), os.path.join(model_dir, 'step.onnx'),
        input_names=['image', 'ids', 'pos'],
        output_names=['logits_row', 'attn_row'],
        opset_version=18, dynamo=True, external_data=False)
    torch.onnx.export(
        aux, (image,), os.path.join(model_dir, 'aux.onnx'),
        input_names=['image'],
        output_names=['shape_logits', 'size_logits', 'color_logits'],
        opset_version=18, dynamo=True, external_data=False)

    for block in model.transformer_blocks:  # leave the model as we found it
        block.attn.store_attn = False
        block.attn.last_attn = None


def check_onnx_parity(model: ToyVLM, model_dir: str, gen: ShapeGenerator) -> None:
    """Torch vs onnxruntime on real (scene, prompt-prefix) steps; asserts
    agreement to fp32 noise before the bundle is considered valid."""
    import onnxruntime as ort

    step = StepModel(model).eval()
    aux = AuxModel(model).eval()
    step_sess = ort.InferenceSession(os.path.join(model_dir, 'step.onnx'))
    aux_sess = ort.InferenceSession(os.path.join(model_dir, 'aux.onnx'))
    tok = model.text_processor.tokenizer

    worst_logit, worst_attn = 0.0, 0.0
    for seed in (7, 42, 4711):
        rng = SceneRandom(seed)
        num = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
        image_np, _ = gen.generate_multi_shape_image(num, False, rng=rng)
        image = image_to_tensor(image_np).unsqueeze(0)

        q_ids = tok.tokenize("how many circles are there")
        prompt = ([tok.bos_token_id] + [tok.img_start_id] + [tok.img_token_id] * NUM_IMG_TOKENS
                  + [tok.img_end_id] + [tok.user_token_id] + q_ids
                  + [tok.assistant_token_id] + [tok.think_start_id])
        ids = torch.tensor([model.text_processor.pad_sequence(prompt, MAX_SEQ_LEN)],
                           dtype=torch.long)
        pos = torch.tensor([len(prompt) - 1], dtype=torch.long)

        with torch.no_grad():
            t_logits, t_attn = step(image, ids, pos)
        o_logits, o_attn = step_sess.run(None, {
            'image': image.numpy(), 'ids': ids.numpy(), 'pos': pos.numpy()})
        worst_logit = max(worst_logit, float(np.abs(t_logits.numpy() - o_logits).max()))
        worst_attn = max(worst_attn, float(np.abs(t_attn.numpy() - o_attn).max()))
        assert int(t_logits.argmax()) == int(np.argmax(o_logits)), \
            f"greedy token flipped between torch and ONNX (seed {seed})"

        with torch.no_grad():
            t_aux = aux(image)
        o_aux = aux_sess.run(None, {'image': image.numpy()})
        for t, o in zip(t_aux, o_aux):
            worst_logit = max(worst_logit, float(np.abs(t.numpy() - o).max()))

    for block in model.transformer_blocks:
        block.attn.store_attn = False
        block.attn.last_attn = None

    print(f"ONNX parity: max |logit diff| {worst_logit:.2e}, max |attn diff| {worst_attn:.2e}")
    # Logits run ~|15|, so 1e-3 absolute is fp32 kernel-order noise; what
    # actually matters -- the greedy token -- is asserted exactly above.
    assert worst_logit < 1e-3 and worst_attn < 1e-5, "ONNX disagrees with torch beyond fp32 noise"


def rng_fixtures() -> dict:
    cases = []
    for seed in FIXTURE_RNG_SEEDS:
        r = SceneRandom(seed)
        u32 = [r.next_u32() for _ in range(16)]
        r = SceneRandom(seed)
        below = [r.randbelow(n) for n in (2, 6, 13, 64, 1000)]
        r = SceneRandom(seed)
        ints = [r.randint(a, b) for a, b in ((1, 12), (8, 12), (16, 22), (28, 35), (0, 63))]
        r = SceneRandom(seed)
        weighted = [r.weighted_choice([0, 1], [17, 3]) for _ in range(12)]
        cases.append({'seed': seed, 'next_u32': u32, 'randbelow': below,
                      'randint': ints, 'weighted_17_3': weighted})
    return {'initseq': PCG_INITSEQ, 'cases': cases}


def scene_fixtures(gen: ShapeGenerator) -> dict:
    cases = []
    for seed in FIXTURE_SCENE_SEEDS:
        rng = SceneRandom(seed)
        num = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
        image, meta = gen.generate_multi_shape_image(num, False, rng=rng)
        raw = image.tobytes()
        cases.append({
            'seed': seed,
            'num_shapes_requested': num,
            'metadata': meta,
            'image_sha256': hashlib.sha256(raw).hexdigest(),
            'image_rgb_base64': base64.b64encode(raw).decode('ascii'),
        })
    return {'image_size': 64, 'cases': cases}


def transcript_fixtures(model: ToyVLM, gen: ShapeGenerator) -> dict:
    """Golden transcripts from the real generate_response on CPU.

    Attention is stored as a per-token summary (head+layer mean, 64 floats,
    plus argmax cell) for every token, and in full (L,H,64) for the first
    three rationale tokens -- enough to pin the TS capture without bloating
    the file.
    """
    cases = []
    for seed, question in FIXTURE_QA:
        rng = SceneRandom(seed)
        num = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
        image_np, _ = gen.generate_multi_shape_image(num, False, rng=rng)
        image = image_to_tensor(image_np)
        rationale, answer, details = generate_response(
            model, image, question, return_details=True, return_attention=True)

        def attn_summary(attn_list):
            out = []
            for a in attn_list:  # (L,H,64) tensors
                mean = a.mean(dim=(0, 1))
                out.append({'mean': [round(float(v), 6) for v in mean],
                            'argmax': int(mean.argmax())})
            return out

        cases.append({
            'seed': seed,
            'question': question,
            'rationale': rationale,
            'answer': answer,
            'rationale_tokens': [[w, round(p, 6)] for w, p in details['rationale_tokens']],
            'answer_tokens': [[w, round(p, 6)] for w, p in details['answer_tokens']],
            'answer_topk': [[w, round(p, 6)] for w, p in details['answer_topk']],
            'aux': {fam: [[name, count, round(prob, 6)] for name, count, prob in rows]
                    for fam, rows in details['aux'].items()},
            'rationale_attn_mean': attn_summary(details['rationale_attn']),
            'answer_attn_mean': attn_summary(details['answer_attn']),
            'rationale_attn_full_first3': [
                [[[round(float(v), 6) for v in head] for head in layer] for layer in a]
                for a in details['rationale_attn'][:3]],
        })
    return {'cases': cases}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the Toy VLM web bundle.")
    ap.add_argument('--checkpoint', type=str, default='toy_vlm_cot.pth')
    ap.add_argument('--vocab', type=str, default='tokenizer_vocab.json')
    ap.add_argument('--out', type=str, default='web')
    args = ap.parse_args()

    torch.manual_seed(0)

    text_processor = TextProcessor()
    text_processor.tokenizer.load_vocab(args.vocab)
    model = ToyVLM(text_processor)
    state = torch.load(args.checkpoint, map_location='cpu')
    ckpt_vocab = state['token_embedding.weight'].shape[0]
    assert ckpt_vocab == text_processor.tokenizer.get_vocab_size(), \
        f"checkpoint/vocab mismatch: {ckpt_vocab} vs {text_processor.tokenizer.get_vocab_size()}"
    model.load_state_dict(state)
    model.eval()

    model_dir = os.path.join(args.out, 'public', 'model')
    fixture_dir = os.path.join(args.out, 'fixtures')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fixture_dir, exist_ok=True)

    gen = ShapeGenerator()

    export_onnx(model, model_dir)
    check_onnx_parity(model, model_dir, gen)

    with open(args.vocab) as f:
        vocab = json.load(f)
    with open(os.path.join(model_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2, sort_keys=True)

    print("Generating fixtures (CPU transcripts take a minute)...")
    fixtures = {
        'rng.json': rng_fixtures(),
        'scenes.json': scene_fixtures(gen),
        'transcripts.json': transcript_fixtures(model, gen),
    }
    for name, data in fixtures.items():
        with open(os.path.join(fixture_dir, name), 'w') as f:
            json.dump(data, f)
        print(f"  wrote fixtures/{name}")

    tok = text_processor.tokenizer
    heads = model.auxiliary_heads
    manifest = {
        'checkpoint': os.path.basename(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'stats': model_stats(model),
        'constants': {
            'MAX_SEQ_LEN': MAX_SEQ_LEN, 'NUM_IMG_TOKENS': NUM_IMG_TOKENS,
            'IMG_POS_START': IMG_POS_START, 'PREFIX_LEN': PREFIX_LEN,
            'GRID_CELLS': GRID_CELLS, 'MIN_OBJECTS': MIN_OBJECTS,
            'MAX_OBJECTS': MAX_OBJECTS, 'MAX_GEN_LEN': 160,
        },
        'special_tokens': {name: tid for name, tid in tok.SPECIAL_TOKENS.items()},
        'aux_heads': {'shape': heads.shape_names, 'size': heads.size_names,
                      'color': heads.color_names,
                      'num_classes': heads.num_classes},
        'files': {name: sha256_file(os.path.join(model_dir, name))
                  for name in ('step.onnx', 'aux.onnx', 'vocab.json')},
    }
    with open(os.path.join(model_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    for name in ('step.onnx', 'aux.onnx'):
        size = os.path.getsize(os.path.join(model_dir, name))
        print(f"  {name}: {size / 1e6:.1f} MB")
    print(f"Bundle written to {model_dir} and {fixture_dir}")


if __name__ == "__main__":
    main()
