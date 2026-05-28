"""
Optional Mamba-3 baseline loaded from an external source checkout.

The official repository currently exposes Mamba-3 from source. This wrapper
loads ``mamba_ssm/modules/mamba3.py`` without requiring the package initializer
that imports older selective-scan CUDA extensions.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MAMBA3_PATH = "/home/erikg/mamba3"
_MAMBA3_CLASS = None


def _external_root() -> Path:
    return Path(os.environ.get("MAMBA3_PATH", DEFAULT_MAMBA3_PATH)).expanduser().resolve()


def _load_mamba3_class():
    global _MAMBA3_CLASS
    if _MAMBA3_CLASS is not None:
        return _MAMBA3_CLASS

    root = _external_root()
    module_file = root / "mamba_ssm" / "modules" / "mamba3.py"
    if not module_file.exists():
        raise ImportError(
            f"Mamba-3 checkout not found at {root}. "
            "Clone https://github.com/state-spaces/mamba or set MAMBA3_PATH."
        )

    package_name = "mamba_ssm"
    pkg = sys.modules.get(package_name)
    if pkg is None or not hasattr(pkg, "__path__"):
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(root / "mamba_ssm")]
        sys.modules[package_name] = pkg

    module_name = "mamba_ssm.modules.mamba3"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Mamba-3 module from {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    _MAMBA3_CLASS = module.Mamba3
    return _MAMBA3_CLASS


class Mamba3LM(nn.Module):
    """Mamba-3 language model with the same training interface as LadderLM."""

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        d_state=128,
        expand=2,
        headdim=64,
        is_mimo=False,
        mimo_rank=4,
        mamba_chunk_size=16,
        dropout=0.0,
        loss_chunk_size=0,
    ):
        super().__init__()
        Mamba3 = _load_mamba3_class()

        if (dim * expand) % headdim != 0:
            raise ValueError(f"dim * expand must be divisible by headdim: {dim} * {expand} vs {headdim}")

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.loss_chunk_size = loss_chunk_size

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([
            Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=mamba_chunk_size,
                is_outproj_norm=False,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def _keep_kernel_params_fp32(self):
        for layer in self.layers:
            for name in ("dt_bias", "B_bias", "C_bias", "D", "mimo_x", "mimo_z", "mimo_o"):
                param = getattr(layer, name, None)
                if param is not None:
                    param.data = param.data.float()

    def bfloat16(self):
        super().bfloat16()
        self._keep_kernel_params_fp32()
        return self

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
        prev_conv_buffers=None,
        actual_length=None,
        doc_boundaries=None,
    ):
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        x = self.embedding(inp)
        self._keep_kernel_params_fp32()
        for ln, layer in zip(self.layer_norms, self.layers):
            residual = x
            x = ln(x)
            x = layer(x)
            x = residual + x
        x = self.norm(x)

        if return_loss:
            if actual_length is not None:
                device = x.device
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            loss_chunk = self.loss_chunk_size
            if loss_chunk > 0 and x.size(1) > loss_chunk:
                total_sum = x.new_zeros(())
                total_count = 0
                for t0 in range(0, x.size(1), loss_chunk):
                    t1 = min(t0 + loss_chunk, x.size(1))
                    logits_c = self.lm_head(x[:, t0:t1])
                    target_c = target[:, t0:t1]
                    total_sum = total_sum + F.cross_entropy(
                        logits_c.reshape(-1, self.vocab_size),
                        target_c.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_count = total_count + (target_c != -100).sum()
                loss = total_sum / total_count.clamp(min=1)
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            if return_prev_hiddens:
                return loss, (None, None)
            return loss

        logits = self.lm_head(x)
        if return_prev_hiddens:
            return logits, (None, None)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"Mamba3 baseline, dim={self.dim}, depth={self.depth}"


def count_mamba3_params(
    dim,
    depth,
    vocab_size=256,
    d_state=128,
    expand=2,
    headdim=64,
    is_mimo=False,
    mimo_rank=4,
):
    d_inner = dim * expand
    nheads = d_inner // headdim
    rank = mimo_rank if is_mimo else 1
    num_rope_angles = max(1, d_state // 4)
    in_proj_out = 2 * d_inner + 2 * d_state * rank + 3 * nheads + num_rope_angles
    per_layer = (
        dim * in_proj_out
        + 2 * nheads * rank * d_state
        + (3 * nheads * rank * headdim if is_mimo else 0)
        + nheads
        + d_inner * dim
        + 2 * dim
    )
    return vocab_size * dim + depth * per_layer + dim
