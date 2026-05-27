#!/usr/bin/env python3
"""Minimal local generation smoke for pinned v0.1 racer checkpoints.

This script is intentionally small: it loads one local checkpoint, builds the
matching repo model class from the adjacent args.json, runs greedy generation
for a tiny prompt, and emits JSON evidence. It never copies checkpoint weights
or writes large artifacts into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_racer_samples import build_model, load_args, resolve_checkpoint  # noqa: E402


PINNED_CHECKPOINTS = {
    "e88": "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
    "levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt",
    "gdn": "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/"
    "levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt",
    "m2rnn": "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/"
    "levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt",
}


def _module_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _replace_mamba_rmsnorm(norm: torch.nn.Module) -> torch.nn.Module:
    """Replace mamba_ssm's Triton RMSNorm with torch.nn.RMSNorm for CPU."""

    weight = getattr(norm, "weight", None)
    if weight is None:
        return norm
    eps = float(getattr(norm, "eps", 1e-6))
    replacement = torch.nn.RMSNorm(weight.numel(), eps=eps)
    with torch.no_grad():
        replacement.weight.copy_(weight.detach().float())
    return replacement


def _causal_short_conv_cpu(conv: torch.nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """CPU fallback for FLA ShortConvolution with full-context inputs."""

    y = F.conv1d(
        x.transpose(1, 2),
        conv.weight,
        conv.bias,
        padding=conv.kernel_size[0] - 1,
        groups=conv.groups,
    )
    y = y[:, :, : x.size(1)].transpose(1, 2).contiguous()
    activation = getattr(conv, "activation", None)
    if activation in {"silu", "swish"}:
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"unsupported ShortConvolution activation: {activation}")
    return y


def _rms_norm_gated_cpu(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    activation: str,
) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float().view(*([1] * (y.ndim - 1)), -1)
    gate_f = gate.float()
    if activation in {"swish", "silu"}:
        y = y * F.silu(gate_f)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gate_f)
    else:
        raise ValueError(f"unsupported gate activation: {activation}")
    return y.to(dtype=x.dtype)


def _fla_gdn_cpu_core(gdn: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Naive CPU inference path for FLA GatedDeltaNet.

    FLA's official kernels are Triton/CUDA-only in this environment. This
    mirrors the fused recurrent inference recurrence closely enough for a
    loadability smoke while using the checkpoint's actual FLA module weights.
    """

    bsz, seq_len, _ = hidden_states.shape
    q_proj = gdn.q_proj(hidden_states)
    k_proj = gdn.k_proj(hidden_states)
    v_proj = gdn.v_proj(hidden_states)

    if getattr(gdn, "use_short_conv", False):
        q_proj = _causal_short_conv_cpu(gdn.q_conv1d, q_proj)
        k_proj = _causal_short_conv_cpu(gdn.k_conv1d, k_proj)
        v_proj = _causal_short_conv_cpu(gdn.v_conv1d, v_proj)
    else:
        q_proj = F.silu(q_proj)
        k_proj = F.silu(k_proj)
        v_proj = F.silu(v_proj)

    num_heads = int(gdn.num_heads)
    num_v_heads = int(gdn.num_v_heads)
    head_k_dim = int(gdn.head_k_dim)
    head_v_dim = int(gdn.head_v_dim)

    q = q_proj.view(bsz, seq_len, num_heads, head_k_dim)
    k = k_proj.view(bsz, seq_len, num_heads, head_k_dim)
    v = v_proj.view(bsz, seq_len, num_v_heads, head_v_dim)

    if num_v_heads > num_heads:
        repeat = num_v_heads // num_heads
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)

    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()

    beta = torch.sigmoid(gdn.b_proj(hidden_states).float())
    if getattr(gdn, "allow_neg_eigval", False):
        beta = beta * 2.0
    decay_log = -gdn.A_log.float().exp().view(1, 1, num_v_heads) * F.softplus(
        gdn.a_proj(hidden_states).float() + gdn.dt_bias.float().view(1, 1, num_v_heads)
    )

    scale = head_k_dim ** -0.5
    state = hidden_states.new_zeros((bsz, num_v_heads, head_k_dim, head_v_dim), dtype=torch.float32)
    outputs = []
    for t in range(seq_len):
        state = state * torch.exp(decay_log[:, t]).view(bsz, num_v_heads, 1, 1)
        k_t = k[:, t]
        q_t = q[:, t] * scale
        v_t = v[:, t]
        beta_t = beta[:, t].unsqueeze(-1)
        predicted = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta_v = beta_t * (v_t - predicted)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta_v)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q_t))

    out = torch.stack(outputs, dim=1).to(dtype=hidden_states.dtype)
    if getattr(gdn, "use_gate", False):
        gate = gdn.g_proj(hidden_states).view(bsz, seq_len, num_v_heads, head_v_dim)
        out = _rms_norm_gated_cpu(
            out,
            gate,
            getattr(gdn.o_norm, "weight", None),
            float(getattr(gdn.o_norm, "eps", 1e-5)),
            getattr(gdn.o_norm, "activation", "swish"),
        )
    else:
        out = F.rms_norm(
            out,
            (head_v_dim,),
            weight=getattr(gdn.o_norm, "weight", None),
            eps=float(getattr(gdn.o_norm, "eps", 1e-5)),
        )
    return gdn.o_proj(out.reshape(bsz, seq_len, num_v_heads * head_v_dim))


def _fla_layer_cpu_forward(self: torch.nn.Module, x: torch.Tensor, h0=None, **kwargs):
    del h0, kwargs
    output = _fla_gdn_cpu_core(self.gdn, x)
    return self.dropout(output), None


def install_cpu_fallbacks(model: torch.nn.Module) -> list[str]:
    fallbacks: list[str] = []
    if hasattr(model, "fused_add_norm"):
        model.fused_add_norm = False
        fallbacks.append("disabled LadderLM fused_add_norm")
    if hasattr(model, "layer_norms"):
        for i, norm in enumerate(model.layer_norms):
            if norm.__class__.__module__.startswith("mamba_ssm."):
                model.layer_norms[i] = _replace_mamba_rmsnorm(norm)
                fallbacks.append(f"replaced mamba_ssm RMSNorm layer_norms[{i}]")
    if hasattr(model, "norm") and model.norm.__class__.__module__.startswith("mamba_ssm."):
        model.norm = _replace_mamba_rmsnorm(model.norm)
        fallbacks.append("replaced mamba_ssm final norm")

    try:
        from ndm.models.fla_gated_delta import FLAGatedDeltaNetLayer
    except Exception:
        FLAGatedDeltaNetLayer = None
    if FLAGatedDeltaNetLayer is not None:
        count = 0
        for module in model.modules():
            if isinstance(module, FLAGatedDeltaNetLayer):
                module.forward = types.MethodType(_fla_layer_cpu_forward, module)
                count += 1
        if count:
            fallbacks.append(f"installed Python CPU FLA-GDN forward on {count} layers")
    return fallbacks


def unique_layer_classes(model: torch.nn.Module) -> list[str]:
    if hasattr(model, "layers"):
        return sorted({_module_name(layer) for layer in model.layers})
    return []


def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
    cpu_dtype: torch.dtype,
    cuda_dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], list[str]]:
    model_args = load_args(checkpoint)
    tokenizer_name = model_args.get("tokenizer")
    if tokenizer_name:
        import tiktoken

        vocab_size = tiktoken.get_encoding(tokenizer_name).n_vocab
    else:
        vocab_size = 256

    model = build_model(model_args, vocab_size)
    ckpt = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_meta = {"step": ckpt.get("step"), "loss": ckpt.get("loss")}
    del ckpt

    fallbacks: list[str] = []
    if device.type == "cpu":
        fallbacks = install_cpu_fallbacks(model)
        model = model.to(dtype=cpu_dtype)
    else:
        model = model.to(device=device, dtype=cuda_dtype)
    return model.eval(), model_args, ckpt_meta, fallbacks


def encode_prompt(tokenizer_name: str | None, prompt: str):
    if tokenizer_name:
        import tiktoken

        enc = tiktoken.get_encoding(tokenizer_name)
        return enc.encode(prompt, disallowed_special=()), enc.decode, enc.n_vocab
    prompt_bytes = list(prompt.encode("utf-8"))
    return (
        prompt_bytes,
        lambda toks: bytes([t for t in toks if 0 <= t < 256]).decode("utf-8", errors="replace"),
        256,
    )


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    decode,
    device: torch.device,
    max_new_tokens: int,
    max_context: int,
) -> dict[str, Any]:
    generated = list(prompt_tokens)
    finite_checks: list[bool] = []
    argmax_scores: list[float] = []
    for _ in range(max_new_tokens):
        ctx = generated[-max_context:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x, return_loss=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        finite = bool(torch.isfinite(logits).all().item())
        finite_checks.append(finite)
        next_logits = logits[0, -1].float()
        next_token = int(torch.argmax(next_logits).item())
        argmax_scores.append(float(next_logits[next_token].item()))
        generated.append(next_token)

    new_tokens = generated[len(prompt_tokens) :]
    return {
        "generated_token_ids": generated,
        "generated_new_token_ids": new_tokens,
        "generated_text": decode(generated),
        "generated_new_text": decode(new_tokens),
        "all_logits_finite": all(finite_checks),
        "finite_checks": finite_checks,
        "argmax_scores": argmax_scores,
    }


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PINNED_CHECKPOINTS), required=True)
    parser.add_argument("--checkpoint", default=None, help="Override the pinned checkpoint path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--prompt", default="The theorem states")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--cuda-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.checkpoint or PINNED_CHECKPOINTS[args.model])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    started = time.time()
    model, model_args, ckpt_meta, fallbacks = load_checkpoint_model(
        checkpoint=checkpoint,
        device=device,
        cpu_dtype=parse_dtype(args.cpu_dtype),
        cuda_dtype=parse_dtype(args.cuda_dtype),
    )
    load_elapsed_s = time.time() - started

    tokenizer_name = model_args.get("tokenizer")
    prompt_tokens, decode, vocab_size = encode_prompt(tokenizer_name, args.prompt)
    gen_started = time.time()
    generation = greedy_generate(
        model=model,
        prompt_tokens=prompt_tokens,
        decode=decode,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
    )
    generate_elapsed_s = time.time() - gen_started

    new_text = generation["generated_new_text"]
    generated_nonempty = len(generation["generated_new_token_ids"]) > 0 and len(new_text) > 0
    ok = bool(generated_nonempty and generation["all_logits_finite"])

    result = {
        "ok": ok,
        "label": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt_meta["step"],
        "checkpoint_loss": ckpt_meta["loss"],
        "args_json": str(checkpoint.parent / "args.json"),
        "device": args.device,
        "torch_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "model_class": _module_name(model),
        "layer_classes": unique_layer_classes(model),
        "model_num_params": int(model.get_num_params()) if hasattr(model, "get_num_params") else None,
        "level": model_args.get("level"),
        "dim": model_args.get("dim"),
        "depth": model_args.get("depth"),
        "n_heads_arg": model_args.get("n_heads"),
        "n_state": model_args.get("n_state"),
        "expansion": model_args.get("expansion"),
        "use_triton": model_args.get("use_triton"),
        "use_gate": model_args.get("use_gate"),
        "gate_activation": model_args.get("gate_activation"),
        "tokenizer": tokenizer_name or "byte",
        "vocab_size": vocab_size,
        "prompt": args.prompt,
        "prompt_token_ids": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "cpu_fallbacks": fallbacks,
        "load_elapsed_s": load_elapsed_s,
        "generate_elapsed_s": generate_elapsed_s,
        **generation,
        "generated_nonempty": generated_nonempty,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
