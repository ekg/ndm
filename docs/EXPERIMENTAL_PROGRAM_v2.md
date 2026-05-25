# Experimental Program v2 — Next-Round Memo

**Audience:** team lead, planning the next training round (~1 month out).
**Author:** v19-followup-design (Architect).
**Status:** forward-looking design memo. No code, no training runs, no
paper edits in this artifact. The current paper (v18, commit `8f4830c`;
v19-perspective in flight) is the public-facing artifact; this memo
scopes what replaces and extends it once the next-round 1.27–1.35 B
models train to chinchilla-optimal token budgets.

**Scope of decisions made here.** Token budget per architecture, seed
count, checkpoint cadence, which baselines are added, which plots are
refreshed, which experiments belong in v2 of the paper versus a
follow-up. Where a decision depends on information not yet verified
(notably Mamba-3 reference-implementation availability), the memo says
*need-to-verify* rather than asserting.

**Baseline anchor.** The v18 racer (`paper/main.typ` §5, `@fig_lm_racers`)
is a mid-training snapshot:

| Model       | Step at snapshot | Tokens seen (approx) | Loss   |
|-------------|------------------|----------------------|--------|
| Emender     | 1,035,000        | ~10.6 B              | 2.66   |
| GDN         | 1,371,000        | ~11.2 B              | 2.68   |
| M²RNN-CMA   | 958,000          | ~9.8 B               | 2.77   |
| Mamba2      | 1,722,000        | ~14.1 B              | 2.58   |

(Tokens computed from `steps × batch × 2048` using per-architecture
batches in `paper/main.typ` §5 setup table. Mamba2 is in the QA panel
data — `docs/QA_REASONING_PROGRESSION.md` — but is not in the §5 racer
in the paper body; that decision is revisited in §1 below.)

None of these runs are chinchilla-optimal yet; the §1 retrain is the
correct mid-training-snapshot replacement that the v18 paper's
limitations section (`paper/main.typ`, "Snapshot status of the 1.27 B
racer") explicitly defers.

---

## §1 — Chinchilla-optimal retrain protocol

### Token budget per model

Chinchilla scaling (Hoffmann et al. 2022) is compute-optimal at
*~20 training tokens per model parameter* — the headline 20:1 ratio. At
the production parameter band:

- Emender 1.273 B → **25.5 B tokens** (≈12.5 M optimization steps at
  batch 5, ctx 2048 → ~10,240 tokens/step).
- M²RNN-CMA 1.307 B → **26.1 B tokens**.
- GDN 1.352 B → **27.0 B tokens**.

We will treat **26 B tokens** as the common nominal budget, with the
per-architecture exact figure used inside (a) the LR schedule's total
horizon and (b) the chinchilla-optimal reporting label. Token budgets
above 20:1 (e.g. Llama-class 60–100:1) cost more compute for diminishing
returns on the racer band-match claim and are out of scope for the
v2 paper round; they belong to a separate "extended training" follow-up.

The v18 paper's snapshot data corresponds to ~8–14 B tokens per model
(38–55 % of the chinchilla target). v2-paper closes the remaining
~50 % of the curve.

### Models in scope

Mandatory:
1. **Emender (E88)** — `ndm/models/e88_fused.py`, the paper's headline.
2. **M²RNN-CMA** — the raw-write PNR baseline (the within-class
   contrast).
3. **Gated DeltaNet (FLA implementation)** — the linear-recurrent
   frontier baseline.

Recommended additions:
4. **Transformer baseline (small Llama-class at 1.3 B)** — a frontier
   *non-recurrent* anchor. A reviewer of v2 will (correctly) ask "what
   does a same-budget transformer look like?" The current paper sidesteps
   this by being a within-recurrent comparison; v2 should plant the
   transformer reference point even if the headline framing remains
   recurrent-internal, because the *class-level claim* in §1 of the paper
   ("nonlinearity in time is not a cost") is incomplete without a
   transformer datum. **Architecture choice:** the Llama-3.2-1B
   configuration (dim 2048, depth 16, GQA, RoPE) is the de-facto small
   transformer recipe with public scaling data, and serves as the
   "matched-budget, different-class" anchor. Use the same Pile data,
   same tokenizer, same optimizer (schedule-free AdamW), same context
   (2048). This is *one extra run* of the same shape as the four already
   planned and is worth the cost.
5. **Mamba2** — already in the QA panel
   (`docs/QA_REASONING_PROGRESSION.md`) and at step 1.72 M in the
   provenance data; **promote it from "QA only" to a full §5 racer
   row** in v2. It is the canonical selective-SSM baseline; its absence
   from the v18 racer body is a holdover from training-progress reasons,
   not a principled exclusion (see `NEXT_STEPS.md` W4 closure path).

Conditional addition:
6. **Mamba-3** — *need to verify*. The task brief points at Lahoti et
   al., ICLR 2026 / arXiv March 2026. Treated in §3 below; we do not
   commit a row here until availability is confirmed.

**Total v2-paper racer:** 5 mandatory rows (Emender, M²RNN-CMA, GDN,
small-Llama, Mamba2), optionally 6 with Mamba-3 if §3 unblocks.

### Hardware budget estimate

The v18 round provided ~14 B tokens for the longest-running model in
~10–15 wall-days at the genomics-institute cluster's per-model
throughput (read off the snapshot table: Mamba2 1.72 M steps in the
observation window).

For the chinchilla-optimal 26 B-token runs:
- **~25–28 GPU-days per 1.3 B-class recurrent model** (Emender, GDN,
  Mamba2, M²RNN-CMA), with per-architecture throughput differences
  absorbed into the wall-clock figure (linear-state architectures
  parallelise the time axis and finish faster per step; nonlinear-state
  architectures use the multi-programmed wide kernel and finish faster
  per epoch on the same hardware — the §5 racer empirically confirms
  the wall-clock band match).
- **~30 GPU-days for the Llama-1B transformer**, dominated by
  attention's O(T²) at ctx 2048 (a transformer at this scale on Pile is
  the most-studied recipe in the field; ~30 GPU-days for 26 B tokens at
  1.3 B is consistent with the Hoffmann et al. and Llama-3 scaling
  reports).

**Aggregate:** 5 × ~28 = ~140 GPU-days serial. With 2-way pipelining on
the cluster (typical), this is **~10 calendar weeks**. With 3-way (if
the cluster supports it), ~7 weeks. The v18-paper budget actually
accommodated 4 concurrent runs at lower per-model token budget; this
round's chinchilla-optimal scope-up is the only major delta.

If a Mamba-3 row is added (§3), budget another ~25–30 GPU-days.

### Seed strategy

The v18 paper is **single-seed per architecture at 1.27–1.35 B**
(`paper/main.typ` Limitations: "additional seeds and architecture-internal
revalidation"). For v2, the affordable upgrade is:

- **Two seeds for Emender and GDN** (the two architectures in the
  loss-band match — the most-load-bearing within-paper claim). Two seeds
  is enough to put a per-seed band on the racer panel and is not enough
  to make sweeping variance claims, but it is sufficient to defend
  against the "single-seed coincidence" reviewer ask.
- **Single seed for M²RNN-CMA, Mamba2, small-Llama** (the role of these
  rows is anchor / context, not within-class contrast). The §6 8 M
  expressivity probes already use 3-seed protocol; that survives
  unchanged.

Total cost of two-seed treatment for two models: +2 × ~28 ≈ +56
GPU-days, raising the aggregate to ~200 GPU-days serial. This is the
single largest cost lever in §1 and the one most worth paying — a
single-seed racer is the most cited weakness of the v18 paper.

### Snapshot / checkpoint cadence

- **Checkpoint every 1 B tokens** (≈ every 100 K steps at batch 5 × ctx
  2048). That gives 26 checkpoints per run — the right density for the
  smoothed loss-vs-wallclock curve and for periodic QA-panel evaluation.
- **Hot-snapshot every 6 hours** for live mid-training observability
  (the `scripts/run_periodic_racer_evals.py` pattern, already in place).
  Used for the periodic QA panel; not used for archival.
- **Archive every 4 B tokens** (≈ 4 checkpoints retained per run after
  ablation cleanup) plus the final-step checkpoint, plus the
  chinchilla-optimal (20:1) checkpoint if it does not coincide with a
  4 B-token snapshot. Disk-friendly without sacrificing the curve.

### Plots and tables in the current paper that get replaced

- **§5 Table** ("Model / Params / Batch / Shape") — the parameter and
  shape columns survive unchanged; the *Batch* column may shift for the
  small-Llama row. No regenerated artifact, just an added row.
- **`@fig_lm_racers` (Figure 2, loss vs wallclock at the snapshot)** —
  *replaced* by the chinchilla-optimal full-curve panel. Same panels
  (A: log-wallclock; B: tail). New rows: small-Llama, Mamba2
  (promoted), optionally Mamba-3.
- **§6 "QA and reasoning panel at 1.27 B" subsection** — refreshed at
  final-checkpoint, not mid-training; the within-noise statement in
  v18 should be revisited and may either tighten (final separation
  emerges) or stay (the paragraph stands).
- **Snapshot status / "in-progress" caveats throughout v18 §5 and
  §Limitations** — *removed*. The "training in progress" language goes
  away in v2.

The §6 8 M expressivity probes (`@fig_s5_bars`, `@tab_s5`) **do not
change**; those numbers are already at 3-seed protocol at the 8 M
parameter-matched scale and are not load-bearing on the chinchilla-optimal
retrain. The Lean theorem set, the §3 ablation table, the §4 systems
material, and the §7 formal results all carry forward unchanged.

### What is *not* in §1's scope

- Architecture-internal revalidation (output-gate ablation, tanh-vs-
  linear-on-state, decay parameterisation) at 1.27 B. Carried as
  v18-paper limitation; defer to a separate ablation paper or to the
  v2-paper appendix only if budget allows.
- Larger-scale runs (3 B, 7 B). Out of scope for v2; roadmap-only.

---

## §2 — Fine-tuning regime for reasoning evaluation

### Motivation

The v18 paper's 8 M probes show a clean Emender-vs-baseline separation
on S₅/S₃ (`@tab_s5`); the 1.27 B QA panel shows all three architectures
within one standard error of one another on a 300-item NLP panel and at
near-random on BBH/ReCLor/FOLIO multi-step reasoning
(`docs/QA_REASONING_PROGRESSION.md`). The natural follow-up is to ask
whether the small-scale expressivity advantage **manifests at the
chinchilla-optimal scale on reasoning tasks that genuinely require
state-tracking** — once each model has seen enough tokens to be
instruction-followable and reasoning-attempting.

### Benchmark selection

Three tiers, ordered by what is most diagnostic for the Emender's
positioning:

**Tier 1 — synthetic state-tracking at scale (highest signal).**
- **S₅-extended.** Re-run the §6 probe at training lengths T ∈ {1024,
  2048, 4096} on each chinchilla-optimal foundation model after FT on
  a *synthetic-only* mix (transposition-composition strings labeled with
  prefix products). This is the cleanest reading of "does the small-scale
  S₅ separation survive scaling and instruction-style fine-tuning?"
- **State-tracking subsets of BIG-Bench Hard.** Specifically
  `tracking_shuffled_objects_{3,5,7}_objects` and
  `logical_deduction_{3,5,7}_objects`. The v18 panel shows all
  architectures near-random on the 7-object versions
  (`docs/QA_REASONING_PROGRESSION.md`); a separation at 5- or 7-objects
  post-FT would be the headline.

**Tier 2 — mainstream reasoning benchmarks (publication anchor).**
- **GSM8K** (math word problems) — 8.5 K train, 1 K test. Multi-step
  arithmetic chain. The most-cited LLM reasoning benchmark; reviewers
  will expect a row.
- **BBH** (BIG-Bench Hard, 23 tasks). Already partially in the v18 panel.
  Run the full suite post-FT.
- **AGIEval LSAT-LR and LSAT-AR** (the logical-reasoning slices) —
  multi-step deduction where state-tracking would help.

**Tier 3 — context/coverage (one row, not load-bearing).**
- **MMLU-STEM** — broad knowledge floor check, not state-tracking-
  specific, included so the panel matches what readers expect.

**Explicitly excluded:** open-ended generation evals (MT-Bench,
AlpacaEval), code-generation evals (HumanEval, MBPP). These do not bear
on the state-tracking claim and would dilute the message. AGIEval's
math slice overlaps GSM8K — keep GSM8K, skip the math slice.

### Fine-tuning protocol

**Two-stage protocol** balancing cost against headline strength:

**Stage A — LoRA on each foundation model**, on a mixed
instruction+reasoning corpus. Rank 16 LoRA, applied to the recurrent
projection matrices (for the Emender / M²RNN-CMA / Mamba2) or to
attention QKV (for the small-Llama). Schedule: 1 epoch over ~5 B tokens
of FT data, cosine LR, ~1 day per model. **Purpose: rank the four
architectures cheaply and surface separations to focus Stage B on.**

**Stage B — full SFT on the LoRA-winning subset**, on the same data,
LR ~2 e-5 with linear decay. ~5 B tokens of FT data, 2–3 days per model.
**Purpose: produce the publishable separation numbers under a
recipe the field will accept as comparable.**

**Both stages on chinchilla-optimal checkpoints.** No FT on mid-training
snapshots.

### Instruction-tuning dataset

**Tulu V3 mixture** (Lambert et al. 2024 / AllenAI) — well-curated,
includes diverse reasoning (MetaMath, NuminaMath), code (OpenCoder),
and instruction-following (Open-Orca subset). Approximately 1 M
instances. The mixture is published and reproducible. Alternative is
Open-Instruct (older) or a custom mix; Tulu V3 is preferred for
provenance reasons (one named source, public).

For the S₅-extended Tier-1 probe: a **separate synthetic-only FT** of
~100 M tokens of S₅/S₃ transposition strings labeled with prefix
products. Run *after* Tulu V3 FT (so the model retains instruction-
following ability). This is the head-to-head state-tracking probe at
1.27 B scale that the v18 paper currently lacks.

### Evaluation harness

**`lm-evaluation-harness`** (EleutherAI) — the standard, used by every
recent open-weights paper. Pinned to a tagged version with the v2-paper
result table.

- BBH, MMLU, GSM8K, AGIEval: lm-eval-harness configurations exist.
- S₅-extended: custom harness in `experiments/expressivity_tasks/`
  (already exists for the 8 M probes; extend to the FT checkpoint
  loading path).

For the tracking subsets of BBH and the BBH `tracking_shuffled_objects`
tasks, evaluate at the standard 3-shot setting and report mean over
seeds; lm-eval-harness supports both.

### Compute estimate

Per architecture:
- Stage A LoRA FT: ~1 GPU-day.
- Stage B full SFT: ~3 GPU-days.
- Synthetic S₅-extended FT: ~1 GPU-day.
- Full eval suite (BBH + MMLU + GSM8K + AGIEval + S₅-extended): ~0.5
  GPU-days.

Per architecture aggregate: ~5–6 GPU-days. **Five architectures total
(if small-Llama and Mamba2 included): ~25–30 GPU-days, ~3–4 calendar
weeks** if pipelined 2-way.

### Publishable-separation threshold

A finding is publishable as a separation result if:
1. Emender's accuracy on at least one Tier-1 benchmark exceeds the best
   recurrent baseline (GDN, Mamba2, M²RNN-CMA) by **≥ 5 percentage
   points** (mean over 2 seeds on the chinchilla-optimal Emender vs
   single-seed on others, with explicit error bars).
2. The separation is sustained on at least one Tier-2 benchmark
   (GSM8K, BBH, or AGIEval-LSAT) by ≥ 3 pp.
3. The synthetic-S₅-extended FT row shows monotonic Emender lead at
   T = 1024, 2048, 4096, with the gap not collapsing into noise at
   the longest length.

A finding consistent with "no separation" — all four recurrent
architectures within ~2 pp on every benchmark — is **also publishable**,
as a negative result that bounds the practical scope of the §6
expressivity claim. The follow-up paper's framing depends on which
outcome falls out; see §6 below.

### What §2 does *not* commit to

- RLHF or DPO. Out of scope; the state-tracking signal does not depend
  on alignment-style tuning. If the separation does appear, RLHF can
  follow in a separate paper.
- Larger FT context (8 K, 32 K) than the 2 K pretraining context.
  Length-extrapolation behaviour at FT time is interesting but is one
  more confounder; keep FT context at 2 K and treat
  S₅-extended at T > 1024 as the dedicated long-context probe.

---

## §3 — Mamba-3 baseline feasibility

### What needs verification

The task brief cites *Lahoti et al., ICLR 2026 / arXiv March 2026* as
the Mamba-3 reference. As of the v18 paper artifacts, the repo does
**not** carry a Mamba-3 implementation; the related-work catalog
(`docs/related_work_nonlinear_rnns.md`) covers Mamba and Mamba2 but not
Mamba-3. The first concrete step is verification of the following
facts:

1. **Reference implementation availability.** Is the Mamba-3 codebase
   public (e.g., extending `state-spaces/mamba` or a new repo)? *Need
   to verify by direct check of arXiv:2603.* — paper PDF — and the
   authors' GitHub.
2. **Released checkpoints.** Does the Mamba-3 release include
   ≥1 B parameter weights on a Pile-class corpus that we can load
   without retraining?
3. **License compatibility.** If the reference impl exists, is its
   license (Apache 2.0, MIT, MosaicML, custom) compatible with our use
   in a comparison paper.

These three points are the gating questions. Verification can be done
in **~half a GPU-day of inspection** plus a ~1-hour read of the paper;
it does not block the §1 retrain protocol from proceeding for the
other four architectures.

### Branches under each verification outcome

**Branch A — reference impl + checkpoint released, license compatible.**
This is the cleanest case. Action:
- Add Mamba-3 as a **probed-only baseline** in v2: load their public
  ≥1 B checkpoint, run the 8 M-class expressivity probes adapted to
  their tokenizer, run the QA panel, run S₅/S₃ at training length and
  T extrapolation. Cost: ~1–2 GPU-days for eval-only.
- Optionally retrain Mamba-3 from scratch at 26 B tokens on our Pile
  pipeline for the wallclock racer row. Cost: ~25–30 GPU-days. **Worth
  it if and only if the probe-only result is interesting** (Mamba-3
  comes out either substantially ahead or substantially behind on
  S₅ at training length).

**Branch B — reference impl released, no checkpoint, license OK.**
Train Mamba-3 from scratch at 26 B tokens. ~30 GPU-days. Adds one row
to v2's §5 racer. Worth it for the chinchilla-optimal head-to-head.

**Branch C — paper-only, no public code.** Cite their reported
numbers; do not run a head-to-head. Honesty signal: explicit "we do
not have a reference implementation to retrain in our matched
protocol; the Mamba-3 comparison reduces to citing their reported
S₅ numbers, evaluated under their (not our) training protocol." This
preserves the within-paper consistency at the cost of not being able
to claim a same-protocol comparison.

### Honest assessment

Mamba-3 is **a v2-paper experiment if Branches A or B fire; out of v2
scope if Branch C fires**. The decision tree is short, the
verification cheap, and the prep cost (data pipeline, tokenizer
matching) zero-marginal because the §1 cluster pipeline already
handles Mamba2.

The downside risk of skipping Mamba-3 entirely in v2 is small. The v18
paper does not claim to be Mamba-3-aware and reviewers cannot
reasonably demand the comparison if no public code exists. The upside
of including it (probe-only is enough) is large: Mamba-3 will be the
"is the new linear-state baseline" reviewers ask about, and a
side-by-side S₅ probe with the Emender is the most efficient response.

**Recommendation:** kick off the half-day verification immediately
(Architect to action via separate task) and pursue Branch A or B if
either fires. Do not block §1 on this.

---

## §4 — Hybrid-Emender variant

### Motivation

The v18 paper demonstrates emendation in the pure-recurrent setting and
explicitly frames it as "the cleanest comparison setting, not the
technique's scope" (`paper/main.typ` §1, "Emendation is one response;
hybrids are another"). The natural portability test: drop the
delta-correcting write into the recurrent component of a hybrid
architecture (some attention layers + Emender recurrent layers) and ask
whether the §6 within-PNR ordering (emender > raw-write) survives
outside the pure-recurrent setting.

The v18 §6 "hybrid degradation" result tests `[Emender, Emender, GDN,
GDN]` (AABB) and finds that mixing Emender with *linear-state* blocks
*degrades* state-tracking. The v2 follow-up is the *complementary* test:
mixing Emender with *attention* blocks (which themselves do not have
recurrent state) — does the delta-correcting write transport?

### Smallest meaningful scale

**8 M parameter-matched scale** — same probe shape as the v18 §6
expressivity panel (dim = 384, depth = 4, H = 32, N = 32). Cheap
(~hours per config) and the v18 separation result is already
characterised at this scale, making the hybrid comparison a clean
delta against a known reference.

**100 M stretch** — only if 8 M shows portability. ~1 GPU-day per
config; the test is whether the small-scale finding survives a 10×
scale-up before committing to chinchilla-optimal 1.3 B hybrid runs in
a *third* paper round.

### Topology

Three hybrid layouts, all at depth 4, all at the same parameter count
as the v18 baseline 8 M probe (the 4-layer pure-Emender or pure-attention
configurations match the parameter budget by adjusting `dim` /
attention head count):

1. **Emender ↔ attention sandwich (E-A-E-A).** Alternating.
2. **Emender-majority sandwich (E-E-A-A).** Bottom-heavy nonlinear
   recurrent; top of stack is attention. Mirrors the OLMo-Hybrid /
   Griffin tendency to put attention at the top.
3. **Attention-majority sandwich (A-A-E-E).** Attention at the bottom;
   emendation at the top. Tests whether emendation works as the *final*
   refinement step over an attention-shaped representation.

For each layout, run the v18 §6 canonical six-task sweep (parity,
modular counter K=5, FSM K=4, Dyck-1, associative recall, selective
copy) plus S₅ at T = 128. Three seeds per condition.

The headline contrast for portability is **hybrid-Emender (one of the
above) vs hybrid-raw-write (same topology, M²RNN-CMA update instead of
Emender update) at matched parameter count**. If the v18 within-PNR
ordering ports, hybrid-Emender > hybrid-raw-write on state-tracking; if
it does not, the delta-correcting advantage is specific to the
pure-recurrent setting and the v18 contribution scope is correctly
narrow.

### Reference points (architecture-design inspiration)

- **Merrill, Petty, Sabharwal (2024)** — `merrill2024transformers` in
  the bibliography — the formal foundation for the state-tracking
  argument and the closest "hybrid stacks express things pure recurrent
  cannot" theoretical baseline.
- **M²RNN hybrid configurations** (Mishra et al. 2026, §5.2) —
  matrix-state RNN interleaved with attention at 7 B MoE scale. The
  closest concurrent prior art for the hybrid layout decision; use
  their reported attention:recurrent ratio (~1:3 to 1:5) as a starting
  reference and reposition as the 8 M probe data dictates.
- **OLMo-Hybrid 7B** (`olmohybrid2026`) — pure state-space + attention
  topology, with attention at every Nth layer. Their N = 4 is a
  natural sandwich starting point.
- **Titans / Griffin** — different hybrid recipes, less directly
  comparable but useful as prior-art context for the related-work
  discussion of the v2 paper.

### Compute estimate

- 8 M configs (3 hybrid layouts × 2 update rules × 6 tasks × 3 seeds):
  ~108 small-probe runs at ~5 min each = ~9 GPU-hours total, ~half a
  GPU-day.
- S₅ at the same configs: another ~1 GPU-day.
- 100 M stretch (if 8 M positive): same protocol at 10× cost ≈ ~10
  GPU-days.

**Aggregate ~1.5–11 GPU-days** depending on whether 100 M is run, well
under §1 or §2 cost. This experiment is the cheapest single follow-up
in the v2 program by an order of magnitude.

### What result would establish portability

**Strong portability claim:** hybrid-Emender ≥ hybrid-raw-write on at
least 4 of 6 canonical tasks at 8 M *and* on S₅ at training length, with
the same direction of effect as the v18 §6 pure-recurrent ordering
(Emender > raw-write). Two seeds enough; three seeds for safety.

**No portability claim** (also publishable): hybrid-Emender ≈
hybrid-raw-write within one SEM on all tasks. This would establish that
the delta-correcting advantage **requires** the pure-recurrent setting —
which sharpens, not weakens, the v18 contribution by clarifying its
scope.

**Negative portability claim** (most surprising): hybrid-Emender <
hybrid-raw-write on state-tracking. This would suggest emendation
interacts adversely with cross-token mixing, which is a non-trivial
mechanism finding. Less likely *a priori*, but the experiment is cheap
enough to test.

### Sequencing

Hybrid-Emender experiments **do not depend on the chinchilla-optimal
retrain** and can run in parallel with §1. The 8 M probes use the same
codepath as the v18 §6 panel; the only new code is the hybrid stack
assembly. Expect ~1 week elapsed time for the 8 M panel.

---

## §5 — Sequence of experiments

### v2 paper (target: ~2 months after v19 ships)

In scope:
- §1 chinchilla-optimal retrain at 26 B tokens for E88, M²RNN-CMA, GDN,
  Mamba2, small-Llama-1B. Optionally Mamba-3 if §3 unblocks
  (probe-only or full retrain).
- §1.5 refreshed §5 racer figure (`@fig_lm_racers` regenerated) with
  the new rows and full curves. Two-seed bands on E88 and GDN.
- §1.5 refreshed §6 QA-and-reasoning panel at the chinchilla-optimal
  checkpoints. Statement re-read against fresh numbers; "within one
  standard error of one another" stays or is revised.
- §1.5 §3 ablation table unchanged (8 M numbers carry forward).
- §1.5 §7 formal results unchanged.
- §1.5 §9 "Snapshot status of the 1.27 B racer" deleted; equivalent
  reproducibility text added.

Out of v2 scope (deferred to follow-up paper):
- §2 fine-tuning regime.
- §4 hybrid-Emender.

### Follow-up paper (target: ~4–6 months after v19 ships)

Working title: *"Does the Emender's expressivity advantage transfer?"* or
similar. Possible scope:
- §2 fine-tuning reasoning panel (Tulu V3 + lm-eval-harness, all
  architectures, LoRA + full SFT on the v2 chinchilla-optimal
  checkpoints). Publishable-separation criteria from §2 above.
- §4 hybrid-Emender at 8 M and (conditionally) 100 M.
- §4-stretch: if §4-8M shows portability, a single 100 M hybrid-Emender
  run trained for ~10 B tokens, as a "feasibility at scale" rather than
  a full racer.

The follow-up paper's framing depends on the §2/§4 outcomes; see §6.

### Roadmap-only (>6 months out)

- Scale beyond 1.3 B (3 B chinchilla-optimal would require ~60 B tokens
  × ~60 GPU-days per model, ~5–10× the v2 budget; out of cluster scope
  in 2026). Cited in v2's Future Work, not pursued.
- Formal multi-step S₅ inseparability (the Lean frontier
  `formal/lean/LEAN_FRONTIER.md`).
- ParaRNN-style time-axis parallelisation applied to the Emender update.
- 3 B/7 B hybrid-Emender at chinchilla-optimal token count.

---

## §6 — Risks and contingencies

For each risk: probability assessment (rough — informed by v18
snapshot trajectories and the published behaviour of comparable
architectures); what we learn either way; how the contribution shifts.

### Risk A — band-match against GDN does *not* survive chinchilla-optimal scaling

**Probability:** ~25 %. The v18 snapshot shows Emender ahead of GDN by
0.02 nats at fewer tokens (Emender 2.66 at ~10.6 B; GDN 2.68 at ~11.2 B).
The trend through the sampled window is leadership *trading* between
the two, not monotone divergence. The most plausible failure mode is
GDN's larger parameter count (1.352 B vs 1.273 B) and larger token
count (chinchilla-optimal 27 B vs 25.5 B) pulling its loss strictly
below Emender's at the chinchilla-optimal endpoint.

**What we learn either way:**
- If band-match holds: v2's class-level claim (§1 of the paper —
  "nonlinearity in time is not a cost") strengthens; the v18 framing
  is exactly right.
- If GDN pulls ahead by >0.05 nats: the class-level claim weakens to
  "PNR is *within* the loss band of frontier linear-recurrent at the
  chinchilla budget tested" — still a substantive result (rules out
  the strongest form of the "PNR cannot scale" verdict), but not the
  parity headline.

**Fallback story:** re-frame the v2-paper headline around the §6
expressivity separation and the §7 Lean separation, not around the §5
loss band. Both are unaffected by the §5 outcome. The §6 8 M results
and the §7 formal results carry the within-class contrast on their own.
The §5 chinchilla-optimal panel becomes "PNR is competitive on
language-modelling loss, within ~5–10 % of frontier linear-recurrent
baseline at matched chinchilla budget" — still publishable, less
quotable.

**How contribution changes:** small. The v18 paper's contribution is a
synthesis (§1), and the loss-band-match is one of five components, not
the keystone.

### Risk B — fine-tuning shows no separation across architectures

**Probability:** ~40 %. The v18 QA panel at mid-training is *already*
within noise across architectures, and the v18 paper limits the §6
expressivity claim to small-scale state-tracking probes
(`paper/main.typ` §6, "QA and reasoning panel at 1.27 B: parity-rate
evidence"). The fine-tuning regime would have to either (a) shrink
the noise band, (b) magnify the small-scale state-tracking gap, or
(c) both. None of these is guaranteed; mainstream reasoning benchmarks
at 1.3 B-class scale are notoriously noisy and below the threshold
where reasoning-style improvements show.

**What we learn either way:**
- If separation emerges: the strongest reading of the v18 small-scale
  result — the delta-correcting expressivity *transports* to scale on
  reasoning — is empirically confirmed. This is the headline finding
  of a follow-up paper.
- If no separation: the v18 §6 contribution sharpens to "at the 8 M
  parameter-matched probe scale, the update rule separates; at the
  1.3 B chinchilla-optimal scale on mainstream reasoning, the
  separation is washed out (either by the FT noise floor or by the
  pre-training token budget being insufficient to lift the gap
  above noise)." Still publishable as a *bounded scope* result, and
  still relevant to the field: the synthetic state-tracking probe
  separation is the load-bearing positive evidence; mainstream
  reasoning benchmarks at this scale are noise-limited for everyone.

**Fallback story (no-separation case):** the follow-up paper's frame
becomes "where small-scale state-tracking does and does not transport
to large-scale reasoning." The S₅-extended Tier-1 result becomes
load-bearing here — if S₅-extended at T=1024 *does* show separation
post-FT but mainstream BBH does not, that is itself an interesting
mechanism finding.

**How contribution changes:** the v18 paper's contribution is unchanged.
The follow-up paper's frame shifts from "we demonstrate transport" to
"we delimit transport." Both are publishable; the latter is harder to
sell to a venue's reasoning-benchmark reviewers, easier to sell to a
mechanism-of-expressivity venue.

### Risk C — Mamba-3 wins on S₅ at scale

**Probability:** ~30 %, conditioned on Mamba-3 being available and on
its having state-tracking improvements (its reported design space is
not public to this memo; the probability is intuitive, not derived
from architecture details). Mamba-3 is the most-likely place a
*linear-state* baseline could newly cross the v18 §6 ordering.

**What we learn either way:**
- If Mamba-3 wins S₅ at scale (beats Emender at the FT-scale probe by
  >5 pp): the linear-state ceiling result from Merrill et al. is
  challenged for the specific Mamba-3 design, *or* the v18 §6
  small-scale probe does not generalise to chinchilla-optimal scale in
  the way the paper hints. Either way, the v2 contribution shifts:
  emendation becomes a *technique* that is one path to state-tracking
  expressivity, not the *path*.
- If Mamba-3 ties or loses to Emender on S₅ at scale: the v18 framing
  ("delta-correcting write is the load-bearing differentiator") is
  reinforced and survives the strongest available linear-state
  contender.

**Fallback story (Mamba-3 wins case):** the v2 paper re-frames the
update-rule contribution as **mechanism-of-state-tracking** rather than
**only-path-to-state-tracking**. The Lean separation in §7 is unchanged
— that proof is a one-step *resource separation* between the
delta-correcting and raw-write update rules, not a global claim about
all linear-state architectures. Mamba-3 winning S₅ would say that some
linear-state architectures can also state-track, which is a
*characterisation* of the linear-state class and does not contradict
the within-PNR ordering of §6.

**How contribution changes:** large but not fatal. The v2 paper's
positioning becomes "the delta-correcting write is *one* technique
that enables state-tracking in pure-recurrent matrix RNNs" rather
than "*the* technique." The Lean machinery and the within-class
ordering stay.

### Risk D — hybrid-Emender does not transport

**Probability:** ~35 %. The v18 §6 hybrid-degradation result (AABB
with GDN underperforms pure E88 or pure GDN) is suggestive — mixing
the Emender with linear-scan blocks already degrades. Mixing with
attention may or may not degrade differently. The hybrid-degradation
result is consistent with delta-correcting requiring contiguous
recurrent-state propagation, in which case attention blocks would
disrupt the state-tracking benefit similarly.

**What we learn either way:**
- If hybrid-Emender transports: the v18 contribution generalises to
  hybrid stacks. The follow-up paper has a clean headline.
- If hybrid-Emender does not transport: the v18 contribution scope
  narrows correctly. Publishable as a scope-clarification result.

**Fallback story (no-transport case):** the follow-up paper becomes a
scope-and-mechanism paper rather than a portability paper.

**How contribution changes:** essentially none for the v18 paper's
existing claims; it changes only what the follow-up paper claims, and
in either direction the result is publishable.

### Risk E — operational: the cluster cannot pipeline 5 concurrent 26 B-token runs

**Probability:** ~50 %, depending on cluster availability over the
~10-week window. The v18 round demonstrably ran 4 concurrent models
at lower token budget; the chinchilla-optimal upgrade approximately
doubles the per-model wall time. Whether the cluster can absorb 5
models × 28 GPU-days each in parallel is an operational question.

**What we learn either way:** nothing scientific. This is a logistics
risk, not a scientific one.

**Fallback story:** sequence the runs as **two waves of three**:
- Wave 1: Emender (2 seeds), GDN (2 seeds), M²RNN-CMA (1 seed). The
  three architectures in the v18 paper's headline. ~5 weeks.
- Wave 2: Mamba2 (1 seed), small-Llama-1B (1 seed), optionally Mamba-3
  if §3 unblocks. The added baselines for v2. ~5 weeks.

Total ~10 calendar weeks regardless; the deliverable date is the same.

**How contribution changes:** none; only timing.

### Cross-risk synthesis

The single most important risk is **B (fine-tuning shows no
separation)** — it most directly affects the follow-up paper's
headline. The most likely-to-happen risk is **E (cluster sequencing)**
— affects timing but not science. The most contribution-shifting risk
is **C (Mamba-3 wins)** — affects v2's positioning.

The v18 paper itself is robust to all of A–E except partial cases of
A (large GDN lead) and C (Mamba-3 dominating). Both cases are mitigable
by the §6 / §7 small-scale expressivity claim, which is unaffected by
either failure mode. The v18 paper does not over-promise on either of
these axes and remains defensible under each.

---

## Appendix — Quick-reference cost table

| Workstream                                            | Compute (GPU-days) | Calendar (weeks) | Required for | Status |
|-------------------------------------------------------|---------------------|------------------|--------------|--------|
| §1 — Chinchilla-optimal retrain (5 architectures, 2-seed where listed) | ~200 | ~10 | v2 paper | gated on cluster |
| §3 — Mamba-3 verification (paper read + repo check)   | ~0.5                | ~0.5             | v2 paper §3 decision | none |
| §3 — Mamba-3 probe-only (Branch A)                    | ~2                  | ~0.5             | v2 paper      | gated on §3 verification |
| §3 — Mamba-3 full retrain (Branch B)                  | ~30                 | ~2               | v2 paper      | gated on §3 verification |
| §2 — Fine-tuning panel (5 architectures, LoRA + SFT + S₅-extended) | ~30                 | ~4               | Follow-up paper | gated on §1 finishing |
| §4 — Hybrid-Emender at 8 M (3 layouts × 2 rules)      | ~1.5                | ~1               | Follow-up paper | none |
| §4 — Hybrid-Emender at 100 M (if 8 M positive)        | ~10                 | ~1.5             | Follow-up paper | gated on §4-8 M |

**v2-paper aggregate:** ~200–230 GPU-days, ~10 calendar weeks. Includes
the chinchilla retrain and the Mamba-3 verification + optional probe.

**Follow-up paper aggregate (excluding §2 dependencies on §1):**
~30–40 GPU-days, ~4–5 calendar weeks of dedicated work after the v2
checkpoints exist.

---

## What this memo does *not* commit to*

- No code is changed. No training runs are kicked off by this memo.
  Each workstream listed in §5 is a separate decision the team lead
  approves once v19 ships.
- The Mamba-3 information is **need-to-verify** rather than asserted.
  The first action item out of this memo is the half-day verification
  in §3.
- The fine-tuning separation prediction (§2 publishable-threshold and
  §6 Risk B) is *not* a confidence statement; it is a pre-registered
  criterion that the actual numbers will be measured against.
- This memo is forward-looking. v19-paper is the immediate next public
  deliverable; nothing here changes the v19 scope.

---

*Memo only. Per task brief, no workgraph tasks are created here —
sequencing in §5 is for team-lead review and approval.*
