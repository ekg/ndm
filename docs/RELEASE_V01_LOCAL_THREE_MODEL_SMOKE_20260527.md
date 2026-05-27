# v0.1 Local Three-Model Generation Smoke

Date: 2026-05-27
Task: `release-v01-local-three-model-smoke`

This note records minimal local CPU/GPU load-generation smokes for the pinned
E88, GDN, and M2RNN-CMA checkpoints. Checkpoints were read from their existing
local paths only. No checkpoints, safetensors, HF cache files, Docker layers,
PDFs, token files, or other large generated artifacts were copied into the repo,
staged, committed, uploaded, or published.

## Runtime Evidence

Python/runtime:

- `torch`: `2.9.1+cu128`
- `torch.cuda.is_available()`: `true`
- Visible CUDA device count without masking: `8`
- GPU selected for smoke: physical GPU `4`, exposed to the smoke commands as
  `CUDA_VISIBLE_DEVICES=4` / `cuda:0`
- Pre-smoke GPU evidence for GPU 4:
  `NVIDIA RTX 6000 Ada Generation, 2 MiB / 49140 MiB used, 0% utilization`
- Post-smoke GPU evidence for GPU 4:
  `NVIDIA RTX 6000 Ada Generation, 2 MiB / 49140 MiB used, 0% utilization`

CPU memory/disk evidence before the smoke:

- System memory: `1.0 TiB` total, `937 GiB` available
- Filesystem for repo and `/tmp`: `14T` total, `2.8T` available

JSON transcripts were written outside the repo under:

```text
/tmp/release-v01-local-three-model-smoke-agent-404/
```

The six JSON files total about `24K` and are not staged.

## Smoke Script

The task added a small reusable smoke script:

```bash
python -m py_compile scripts/smoke_local_checkpoint_generation.py
```

The script builds the matching repo model class from each checkpoint's adjacent
`args.json`, loads `checkpoint["model_state_dict"]` with `torch.load(...,
mmap=True)`, runs greedy full-context generation for a tiny prompt, checks that
all logits are finite, and writes JSON evidence. It does not load the
schedule-free optimizer state into an optimizer and does not copy weights.

CPU-specific compatibility:

- `LadderLM` uses `mamba_ssm` Triton RMSNorm when that package is installed; the
  script replaces those norm modules with `torch.nn.RMSNorm` for CPU-only
  forward passes after loading the checkpoint weights.
- The installed FLA GDN kernels are Triton/CUDA-only for CPU tensors. For the
  CPU GDN smoke only, the script runs a naive Python/PyTorch FLA-GDN recurrent
  forward over the loaded FLA module weights. GPU GDN uses the installed FLA
  CUDA/Triton path directly.

## Exact Commands

CPU:

```bash
python scripts/smoke_local_checkpoint_generation.py --model e88 --device cpu --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/e88_cpu.json
python scripts/smoke_local_checkpoint_generation.py --model gdn --device cpu --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/gdn_cpu.json
python scripts/smoke_local_checkpoint_generation.py --model m2rnn --device cpu --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/m2rnn_cpu.json
```

GPU:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_local_checkpoint_generation.py --model e88 --device cuda --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/e88_cuda.json
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_local_checkpoint_generation.py --model gdn --device cuda --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/gdn_cuda.json
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_local_checkpoint_generation.py --model m2rnn --device cuda --prompt "The theorem states" --max-new-tokens 2 --json-out /tmp/release-v01-local-three-model-smoke-agent-404/m2rnn_cuda.json
```

Transcript summary:

```bash
python - <<'PY'
import json, pathlib
base = pathlib.Path('/tmp/release-v01-local-three-model-smoke-agent-404')
for p in sorted(base.glob('*.json')):
    d = json.loads(p.read_text())
    print(p.name, d['ok'], d['label'], d['device'],
          d['generated_new_token_ids'], repr(d['generated_new_text']),
          d['all_logits_finite'])
PY
```

Summary output:

```text
e88_cpu.json True e88 cpu [218, 218] '\x1e\x1e' True
e88_cuda.json True e88 cuda [218, 218] '\x1e\x1e' True
gdn_cpu.json True gdn cpu [318, 318] ' is is' True
gdn_cuda.json True gdn cuda [318, 318] ' is is' True
m2rnn_cpu.json True m2rnn cpu [2109, 34059] '........Officers' True
m2rnn_cuda.json True m2rnn cuda [2109, 34059] '........Officers' True
```

## Results

All smokes used:

- Prompt: `The theorem states`
- Prompt token IDs under `p50k_base`: `[464, 44728, 2585]`
- Seed: `20260527`
- Generation mode: greedy, full-context, `max_new_tokens=2`
- Success condition: nonempty decoded new text and finite logits for every
  generation step

| Model | Device | Status | New token IDs | Decoded new text | Finite logits |
| --- | --- | --- | --- | --- | --- |
| E88 | CPU | PASS | `[218, 218]` | `'\x1e\x1e'` | true |
| E88 | CUDA GPU 4 | PASS | `[218, 218]` | `'\x1e\x1e'` | true |
| GDN | CPU | PASS | `[318, 318]` | `' is is'` | true |
| GDN | CUDA GPU 4 | PASS | `[318, 318]` | `' is is'` | true |
| M2RNN-CMA | CPU | PASS | `[2109, 34059]` | `'........Officers'` | true |
| M2RNN-CMA | CUDA GPU 4 | PASS | `[2109, 34059]` | `'........Officers'` | true |

No NaNs, infinities, tracebacks, or crashes occurred in any of the six runs.

## Model And Checkpoint Details

### E88 / NDM

- Pinned checkpoint:
  `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt`
- Args/config:
  `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/args.json`
- Step/loss loaded from checkpoint: `1281000`, `2.685042345523834`
- Model class: `ndm.models.ladder_lm.LadderLM`
- Recurrent layer class: `ndm.models.e88_fla_hybrid.E88FLAHybrid`
- Tokenizer: `p50k_base`, vocab size `50281`
- Config: `level=E88`, `dim=1664`, `depth=12`, `n_heads=370`,
  `n_state=32`, `expansion=1.0`, `use_triton=1`, `use_gate=1`,
  `gate_activation=silu`
- Parameter count from constructed model: `1,273,191,856`

### GDN

- Pinned checkpoint:
  `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt`
- Args/config:
  `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/args.json`
- Step/loss loaded from checkpoint: `1686000`, `2.6104582071304323`
- Model class: `ndm.models.ladder_lm.LadderLM`
- Recurrent layer class: `ndm.models.fla_gated_delta.FLAGatedDeltaNetLayer`
- Tokenizer: `p50k_base`, vocab size `50281`
- Config: `level=fla-gdn`, `dim=2688`, `depth=21`, `n_heads arg=44`,
  `n_state=64`, `expansion=2.0`, `use_triton=0`, `use_gate=1`,
  `gate_activation=sigmoid`
- Parameter count from constructed model: `1,352,352,498`
- CPU note: GDN CPU used the script's Python fallback for `21` loaded
  `FLAGatedDeltaNetLayer` layers because the installed FLA implementation calls
  Triton kernels for CPU tensors. GPU GDN used the official FLA path.

### M2RNN-CMA

- Pinned checkpoint:
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt`
- Args/config:
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/args.json`
- Step/loss loaded from checkpoint: `1212000`, `2.686988649368286`
- Model class: `ndm.models.m2rnn_baseline.M2RNNLM`
- Recurrent layer class: `ndm.models.m2rnn_baseline.M2RNNLayer`
- Tokenizer: `p50k_base`, vocab size `50281`
- Config: `level=m2rnn`, `dim=1920`, `depth=21`, `n_heads=370`,
  `n_state=16`, `expansion=1.0`, `use_triton=0`, `use_gate=1`,
  `gate_activation=sigmoid`
- Parameter count from constructed model: `1,307,101,140`

## Guardrail Result

Only this small Markdown report and the small smoke script are intended for git.
The local checkpoint directories under `/tmp/pile_convergence_*` remained
read-only inputs. The JSON transcripts stayed under `/tmp` and were not staged.
No HuggingFace or GitHub upload was attempted, no repository visibility was
changed, and no token-bearing paths or large generated artifacts were staged.
