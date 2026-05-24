// Nonlinear Delta Memory — main paper source
// Format: Typst (https://typst.app), NOT LaTeX.
// Build: bash paper/build.sh  →  paper/main.pdf
//
// VOICE CONSTRAINT FOR EVERY SECTION:
//   Write for an outside reader who has never heard of NDM, the Elman
//   lineage, or any internal codenames (E-series, E88, etc.).
//   Introduce every concept on first use; do not assume familiarity.

#set document(
  title: "Nonlinear Delta Memory: Scaling Pure Recurrent Language Models by Multi-Programming",
  author: ("Author Placeholder"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(1em)
  it
  v(0.5em)
}

// ── Bibliography ──────────────────────────────────────────────────────────────
#show bibliography: set heading(numbering: none)

// ── Title block ───────────────────────────────────────────────────────────────
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Nonlinear Delta Memory: Scaling Pure Recurrent Language Models\
    by Multi-Programming
  ]
  #v(0.5em)
  #text(size: 12pt)[Author Placeholder]
  #v(0.25em)
  #text(size: 10pt, style: "italic")[Affiliation Placeholder]
  #v(1em)
]

// ── Abstract ─────────────────────────────────────────────────────────────────
#align(center)[*Abstract*]

// TODO (write for an outside reader; no internal codenames):
// 4–6 sentences covering: (1) the field assumption being rejected
// (pure nonlinear recurrence is impractical at scale), (2) what NDM is
// and its key update equation S = tanh(d·S + k(v − Sᵀk)ᵀ), (3) the
// multi-programming systems insight, (4) the expressivity separation
// result on S5 permutation composition, (5) the trusted Lean 4 core.
#lorem(80)

#v(1em)
#line(length: 100%)
#v(0.5em)

// ── 1. Introduction ───────────────────────────────────────────────────────────
= Introduction

// TODO (write for an outside reader; no internal codenames):
// • State the field assumption to reject: pure serial RNNs are "expressive
//   but impractical at scale"; every billion-scale recurrent LM to date is
//   linear-state or hybrid. Cite representative prior work.
// • Counter-hypothesis: the missing ingredient is multi-programming (many
//   bounded per-head memory programs), not attention or linear scan.
// • Frame the paper's question: did pure nonlinear recurrence fail because
//   it was the wrong computation, or because it was parallelized along the
//   wrong axis?
// • List the six numbered contributions (multi-programming systems +
//   delta-correction mechanism + expressivity results + formal results).
// • Position concurrent work: M2RNN at 410M pure-recurrent and
//   xLSTM-1.3B (mixed sLSTM/mLSTM) are the closest prior comparables.
#lorem(60)

// Placeholder citations to verify bibliography path resolves correctly.
// TODO: move into prose when drafting.
Recent work on structured state-space duality @mamba2_2024 and the delta
rule @deltanet2024 provide linear-state baselines against which we compare.

// ── 2. Background ─────────────────────────────────────────────────────────────
= Background

// TODO (write for an outside reader; no internal codenames):
// • Linear-state recurrence taxonomy: define the linearity criterion
//   h_t = A_t h_{t-1} + b_t with A_t, b_t depending only on x_t. Cover
//   Mamba/Mamba2 @mamba2_2024, DeltaNet @deltanet2024, Gated DeltaNet
//   @gated_deltanet2024, RetNet, GLA, RWKV variants, HGRN, mLSTM, RG-LRU.
// • Nonlinear-state recurrence taxonomy: classical LSTM/GRU (never scaled
//   to ≥500M on Pile-class data), sLSTM (memory mixing), M2RNN (matrix
//   state inside tanh with raw-write injection), Titans, Liquid LFM.
// • State-tracking theory and Barrington witness: solvable vs non-solvable
//   groups; S3 (solvable, 6 states) as control and S5 (non-solvable, 120
//   states) as the NC¹-complete probe.
// • Why matrix state is background, not novelty: d² dynamic-state capacity
//   at O(d²) cost via outer products is the precondition NDM inherits.
#lorem(60)

The state-tracking hardness hierarchy follows from @barrington1986.
Gated delta-rule variants @gated_deltanet2024 are representative
linear-state baselines in our expressivity sweep.

// ── 3. Architecture — Nonlinear Delta Memory ──────────────────────────────────
= Architecture

// TODO (write for an outside reader; no internal codenames):
// • Introduce the architecture as "Nonlinear Delta Memory (NDM)"; describe
//   the per-head update equation:
//     r = Sᵀk;  δ = v − r;  S = tanh(d·S + outer(k, δ));  y = silu(g)·Sᵀq
//   Explain each term for a reader unfamiliar with matrix-state RNNs.
// • Justify each ingredient: (a) tanh-on-state provides nonlinear
//   capacity; (b) the delta write v − Sᵀk is error-correcting and is what
//   separates NDM from raw-write alternatives; (c) many small independent
//   heads expose multi-programming parallelism.
// • Describe the multi-programmed shape used at 1.27B parameters:
//   dim=1664, depth=12, n_heads=370, n_state=32 (370 independent 32×32
//   bounded memory programs per layer per batch element).
// • Contrast with the raw-write alternative (see Related Work): show why
//   the delta correction is the load-bearing ingredient.
#lorem(60)

// ── 4. Systems — Triton Kernel and Multi-Programmed Scaling ───────────────────
= Systems

// TODO (write for an outside reader; no internal codenames):
// • Describe the fused Triton recurrence kernel: one program per
//   (batch, head_block); the state tile lives in registers/SRAM.
// • List fusions: SiLU-q/k/v, L2-q/k normalization, output gate, and the
//   numerically-stable tanh (2·σ(2·x) − 1). Explain why fusing removes
//   GPU kernel-launch overhead and how many milliseconds per step this saves.
// • Explain sparse-checkpoint backward: every CKPT_INTERVAL steps, with
//   the corresponding activation-memory reduction.
// • Explain why Triton (single source for CUDA + ROCm) over hand-tuned
//   HIP: portability at ~70% of hand-tuned throughput in ~1 week.
// • Describe the distributed training plan: ScheduleFree-AdamW per island
//   with hierarchical local-SGD model averaging. Explain why pure-recurrent
//   NDM does not need sequence parallelism.
#lorem(60)

// ── 5. Language Modeling Results ──────────────────────────────────────────────
= Language Modeling Results

// TODO (write for an outside reader; no internal codenames):
// • Describe the experimental setup: byte/token streams, ScheduleFree-AdamW,
//   context-2K curriculum, multi-GPU training configuration.
// • Name the baseline models being compared, introducing each briefly for
//   an outside reader (NDM, a strong linear-gate baseline, Mamba2, and two
//   variants of a concurrent nonlinear-state baseline).
// • Report the headline metric: wallclock loss / bits-per-byte (smoothed
//   last 5K/10K/50K steps). Present Figure 3.
// • Report stability evidence: contrast a divergent baseline (loss and
//   gradient norm vs step) against the stable comparators.
// • Be honest about the gap: note which artifacts are still being staged
//   from out-of-repo training runs.
#lorem(60)

#figure(
  image("figures/figure_3_placeholder.svg", width: 80%),
  caption: [
    *1.27B language-model loss racers.* Loss / bits-per-byte vs wallclock
    for NDM and baselines (smoothed 5K/10K/50K windows).
    // TODO: replace with real loss-curve plot once ~/elman/ artifacts are
    // staged into paper/results/figure_3/
  ],
) <fig_lm_racers>

// ── 6. Expressivity Results ───────────────────────────────────────────────────
= Expressivity Results

// TODO (write for an outside reader; no internal codenames):
// • Explain the permutation-composition probe: S3 (solvable group, 6
//   states) as a control and S5 (non-solvable, 120 states) as the
//   Barrington NC¹-complete probe @barrington1986. Why S5 separates
//   recurrent architectures by computational class.
// • Report the S5/S3 headline numbers at T=128, 3 seeds, 8M parameters
//   (see the table; NDM 0.79 vs linear-state baselines ~0.36 vs
//   concurrent nonlinear-state baseline ~0.22 on S5).
// • Report S5 length extrapolation T=128→256→512→1024.
// • Report the 6-task canonical sweep (parity, modular counter, FSM
//   tracking, dyck, associative recall, selective copy).
// • Report the hybrid ablation: interleaving NDM layers with linear-scan
//   layers degrades state-tracking capability.
// • Interpret: the delta correction, not matrix state nor temporal
//   nonlinearity alone, is the load-bearing mechanism.
#lorem(60)

// ── 7. Formal Results ─────────────────────────────────────────────────────────
= Formal Results

// TODO (write for an outside reader; no internal codenames):
// • Explain the trusted Lean 4 core: what "trusted" means (no sorry /
//   admit / axiom / opaque / native_decide in the proof closure), and why
//   a machine-checked proof matters for separation claims.
// • Theorem set A — update-family resource separation: NDM and the raw-
//   write baseline are provably distinct as update families.
// • Theorem set B — S5 tracker: the S5 group has exactly 120 states and
//   is non-solvable (machine-checked); NDM realizes the full S5 tracker
//   with 480 state/input keys.
// • Theorem set C — finite-state ceiling: fixed-precision recurrent
//   recognizers have finite state spaces; exact lookup-table realization.
// • Theorem set D — capacity separation between the many-head and one-
//   head variants.
// • Explicit non-claims: no Lean lower bound covering all linear-scan
//   models on S5; no Barrington/NC¹ completeness inside Lean; no formal
//   proof that trained real-valued NDM exactly recovers the lookup table.
#lorem(60)

// ── 8. Related Work ───────────────────────────────────────────────────────────
= Related Work

// TODO (write for an outside reader; no internal codenames):
// • Linear-state cohort: Mamba/Mamba2 @mamba2_2024, DeltaNet
//   @deltanet2024, Gated DeltaNet @gated_deltanet2024, RetNet, GLA,
//   RWKV variants, HGRN, mLSTM, RG-LRU/Griffin/RecurrentGemma.
//   Distinguish from NDM by the linearity criterion.
// • Nonlinear-state cohort: sLSTM (xLSTM-1.3B, 7:1 ratio with mLSTM);
//   M2RNN @m2rnn2026 (closest comparable: 410M pure-recurrent, raw-write
//   tanh); Titans (MLP memory with online gradient updates); classical
//   LSTM/GRU (never published at ≥500M Pile-class).
// • Closest-prior-art treatment: explicit M2RNN and xLSTM subsections.
//   State: "To the best of our knowledge, NDM is the first pure nonlinear
//   recurrent language model trained at ≥1B parameters to near-convergence
//   on a large-scale web corpus."
#lorem(60)

// ── 9. Limitations ────────────────────────────────────────────────────────────
= Limitations

// TODO (write for an outside reader; no internal codenames):
// • No completed Lean lower bound for all linear-scan models on S5.
// • S5 length extrapolation degrades at T=1024 (NDM to 0.11).
// • Geometry-sensitivity caveat: strong empirical claims are conditional
//   on the multi-programmed shape, not the update equation alone.
// • Production loss curves and checkpoints are being staged from an
//   out-of-repo training location.
// • The published baseline (raw-write variant) is a sympathetic
//   re-implementation; exact training details may differ.
// • Several architectural contradictions remain open (output gate,
//   tanh vs linear state, decay schedule); production uses conservative
//   settings; revalidation at 1.27B is the highest-value follow-up.
#lorem(40)

// ── 10. Conclusion ────────────────────────────────────────────────────────────
= Conclusion

// TODO (write for an outside reader; no internal codenames):
// • Restate Pillar 1: pure recurrence is viable at billion-parameter scale
//   when shaped as a many-program GPU workload; parallelism is across
//   heads/programs/batch, not along the time axis.
// • Restate Pillar 2: the delta-correcting matrix write is the empirically
//   and formally separable mechanism; nonlinearity alone is not sufficient.
// • Acknowledge what was NOT claimed: no "first nonlinear matrix-state RNN"
//   claim, no NC¹ lower bound, no "linear scans cannot do S5" formal proof.
// • Open horizon: how far nonlinear recurrent reasoning can scale once
//   memory, geometry, and systems are co-designed.
#lorem(40)

// ── Bibliography ─────────────────────────────────────────────────────────────
#bibliography("refs.bib", title: "References", style: "ieee")
