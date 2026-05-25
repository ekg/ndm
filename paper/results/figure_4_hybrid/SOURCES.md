# Figure 4 (hybrid degradation) — per-seed source data

These JSON files are the raw `final_acc` records from the canonical
expressivity sweep (`experiments/expressivity_tasks/run_canonical_sweep.py`)
at the canonical 8 M shape (`dim=384, depth=4, n_heads=32, n_state=32,
sf-AdamW, 10K steps, batch_size=32`).

For each `(pattern, task, seed)` triple the file is the full
training/eval log; the figure script reads `final_acc` only.

Patterns:
- `pure_E88` — 4-layer pure Emender (`[E88, E88, E88, E88]`)
- `pure_FLA` — 4-layer pure Gated DeltaNet (`[fla-gdn, fla-gdn, fla-gdn, fla-gdn]`)
- `hybrid_AABB` — 4-layer Emender+GDN AABB pattern (`[E88, E88, fla-gdn, fla-gdn]`)

Tasks:
- `modular_counter` (K=5 modular counter; sequence length 128)
- `fsm_tracking` (K=4 random FSM; sequence length 256)

Seeds: 42, 123, 456.

Mean ± std agrees with `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`:
| Task            | pure_E88           | pure_FLA           | hybrid_AABB        |
|-----------------|--------------------|--------------------|--------------------|
| modular_counter | 0.903 ± 0.033      | 0.648 ± 0.118      | 0.536 ± 0.238      |
| fsm_tracking    | 1.000 ± 0.000      | 0.830 ± 0.040      | 0.713 ± 0.021      |

Origin tree: `/home/erikg/elman/experiments/expressivity_tasks/results/`.
Mirrored here so `paper/build.sh` is self-contained.
