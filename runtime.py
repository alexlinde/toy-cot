import torch, os
from contextlib import nullcontext


def setup_runtime(prefer_compile=True):
    # Device
    if torch.cuda.is_available():
        device_type, device = "cuda", torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type, device = "mps", torch.device("mps")
    else:
        device_type, device = "cpu", torch.device("cpu")

    # Precision & autocast. bf16 needs no loss scaling, so `scaler` stays None
    # there and both trainers take their plain backward path.
    amp_dtype, scaler = None, None
    autocast_ctx = nullcontext()

    if device_type == "cuda":
        if torch.cuda.is_bf16_supported():          # H100 / L4 and friends
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.conv.fp32_precision = "tf32"      # TF32 convolutions
        torch.backends.cuda.matmul.fp32_precision = "high"     # "ieee" to disable TF32

        # No SDP backend pinning: MultiHeadAttention hands scaled_dot_product_attention
        # a bool prefix-LM mask, which flash attention does not support. Forcing flash
        # on and the fallbacks off would turn every attention call into a hard error.

    elif device_type == "mps":
        amp_dtype = torch.float16
        autocast_ctx = torch.amp.autocast("mps", dtype=amp_dtype)

    def to_device(model):
        model = model.to(device)
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        return model

    def maybe_compile(model):
        if prefer_compile and device_type == "cuda":
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                pass
        return model

    # Dataloader knobs
    cpu_cores = max(1, (os.cpu_count() or 4) - 1)
    if device_type == "cuda":
        num_workers = min(32, cpu_cores)
        pin_memory = True
    elif device_type == "mps":
        num_workers = min(8, cpu_cores)
        pin_memory = False
    else:
        num_workers = min(8, cpu_cores)
        pin_memory = False

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # The one weight-decay policy lives in train_model.build_param_groups, which
    # picks the no-decay set by identity (embeddings, type vectors, pos_scale)
    # rather than by parameter name. Nothing here second-guesses it.
    def make_optimizer_from_groups(param_groups, lr):
        return torch.optim.AdamW(
            param_groups,
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.0,   # groups already specify WD
            fused=True if device_type == "cuda" else False
        )

    return {
        "device": device,
        "autocast": autocast_ctx, "scaler": scaler,
        "to_device": to_device, "maybe_compile": maybe_compile,
        "dataloader_kwargs": dl_kwargs,
        "make_optimizer_from_groups": make_optimizer_from_groups,
    }


