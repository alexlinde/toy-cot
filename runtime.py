import torch, os
from contextlib import nullcontext


def setup_runtime(prefer_compile=True, try_fp8=False, target_eff_batch=None):
    # Device
    if torch.cuda.is_available():
        device_type, device = "cuda", torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type, device = "mps", torch.device("mps")
    else:
        device_type, device = "cpu", torch.device("cpu")

    # Precision & autocast
    amp_dtype, scaler = None, None
    autocast_ctx = nullcontext()

    if device_type == "cuda":
        # H100/L4: prefer bf16; optionally FP8 via TransformerEngine (requires extra code)
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        use_fp8 = False
        if try_fp8:
            # You would need to integrate NVIDIA TransformerEngine to actually use FP8.
            # Leave this False unless you plug TE in your modules.
            use_fp8 = False

        if bf16_ok and not use_fp8:
            amp_dtype = torch.bfloat16
            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            scaler = torch.cuda.amp.GradScaler(enabled=False)  # no scaler needed for bf16
        else:
            amp_dtype = torch.float16
            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        torch.backends.cudnn.benchmark = True

        # Prefer Flash SDP
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

    elif device_type == "mps":
        amp_dtype = torch.float16
        autocast_ctx = torch.amp.autocast(device_type="mps", dtype=amp_dtype)

    # TF32 / precision preferences using new API (CUDA only)
    if device_type == "cuda":
        try:
            # cuDNN convolutions use TF32
            if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
                torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass
        try:
            # cuBLAS matmul uses TF32-equivalent precision
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul") and \
               hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "high"  # or "ieee" to disable TF32
        except Exception:
            pass

    def to_device(model):
        model = model.to(device)
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        return model

    def maybe_compile(model):
        if prefer_compile and device_type == "cuda" and hasattr(torch, "compile"):
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

    def make_optimizer(model, lr, weight_decay=0.01):
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".bias") or "norm" in n.lower() or "ln" in n.lower() or "bn" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
            fused=True if device_type=="cuda" else False
        )
        return opt

    def make_optimizer_from_groups(param_groups, lr):
        return torch.optim.AdamW(
            param_groups,
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.0,   # groups already specify WD
            fused=True if device_type == "cuda" else False
        )

    def linear_scaled_lr(base_lr, base_eff_batch=256, eff_batch=None):
        eff = eff_batch or target_eff_batch or base_eff_batch
        return base_lr * (eff / base_eff_batch)

    return {
        "device": device, "device_type": device_type,
        "autocast": autocast_ctx, "amp_dtype": amp_dtype, "scaler": scaler,
        "to_device": to_device, "maybe_compile": maybe_compile,
        "dataloader_kwargs": dl_kwargs, "make_optimizer_from_groups": make_optimizer_from_groups,
        "make_optimizer": make_optimizer, "linear_scaled_lr": linear_scaled_lr,
    }


