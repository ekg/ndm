# Lean Multi-Step Separation: Frontier Document

**Last updated:** 2026-05-25 (v16-lean-extend, agent-203)

This file documents the state of the multi-step extension of the NΔM vs
fixed-right raw-write M2RNN separation, in Lean. It is a status snapshot, not a
roadmap-with-deadlines.

## What the v15 Lean core proved (baseline)

The v15 Lean core (commit `3553f19`) proves a clean **one-step** resource
separation between NΔM's delta-correcting write and any fixed-weight raw-write
RNN at equal per-token FLOP cost. The headline theorems:

* `ndm_realizes_s5_tracker` — NΔM realizes the S₅ tracker at `d = 12`.
* `ndm_m2rnn_flop_class_equiv` — NΔM and M²RNN-CMA have equivalent per-token
  FLOP cost (both `O(d²)`, within a factor of 2).
* `multiProgrammed_admits_m2rnn_and_ndm` — multi-programming admits both rules.
* `ndm_m2rnn_one_step_resource_separation` — at the witness
  `(lowerLeftState, mixedKey, 0)`, no fixed-right raw-write resource (with
  row/column/cell external forget gate) matches NΔM's mixed-key delta
  correction in one recurrent step.

The empirical claim driving the paper — that NΔM tracks S₅ at 79% while
M²RNN-CMA achieves 22% at training length and the gap widens with sequence
length — concerns full-sequence trajectories. The reviewer's framing was: a
one-step advantage "could in principle wash out over a trajectory or compound;
the proof says nothing about which."

This document records the v16 push past one-step.

## What v16-lean-extend landed

### Milestone 1 — Two-step separation (**clean**)

**File:** `formal/lean/ElmanProofs/Architectures/MultiStepSeparation.lean`

**Theorem:** `ndm_m2rnn_two_step_separation`

```
theorem ndm_m2rnn_two_step_separation :
    twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec)
      = ⟨entry (0,0) = tanh(tanh(-1)), other entries 0⟩
    ∧ ∀ resource : FixedRightRawExternalForget2,
        twoStep resource.update lowerLeftState mixedKey
            (0 : TwoVec) (0 : TwoVec) (0 : TwoVec)
          ≠ twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
              lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec)
```

**Strategy.** Compose the one-step witness with a "zero" step
`(k = 0, v = 0)`. On the M²RNN side, the key lemma
`FixedRightRawExternalForget2_preserves_zero_row` says that any fixed-right
raw-write resource with row/column/cell forget gates preserves the zero-row-0
property of its state when applied with `v = 0`. The witness state
`lowerLeftState` has row 0 = (0, 0). Both step 1 (input `(mixedKey, 0)`) and
step 2 (input `(0, 0)`) preserve the zero row, so entry `(0, 0)` of the
two-step trajectory is exactly `0`. On the NΔM side, step 1 produces entry
`(0, 0) = tanh(-1)` (cross-row delta correction), and step 2 reduces to
elementwise `tanh` (since `(I − 0·0ᵀ) = I`), giving entry
`(0, 0) = tanh(tanh(-1)) ≠ 0` by injectivity of `tanh`.

**Resource class.** Row, column, and cell external forget gates (the same
class covered by the one-step theorem). The scalar-forget case is the row case
specialized to a constant row vector, so it is also covered.

### Milestone 2 — k-step separation (**clean, extending toward full S₅**)

**Theorem:** `ndm_m2rnn_k_step_separation`

```
theorem ndm_m2rnn_k_step_separation (k : ℕ) (hk : 1 ≤ k)
    (resource : FixedRightRawExternalForget2) :
    iterUpdate resource.update lowerLeftState (kStepWitnessInputs k) ≠
      iterUpdate (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState (kStepWitnessInputs k)
```

with `kStepWitnessInputs k = (mixedKey, 0) :: zeroSteps (k - 1)` (length `k`).

**Strategy.** Induct on the tail of the input list. The zero-row-0 preservation
lemma chains across every `(0, 0)` input step; the resulting M²RNN entry
`(0, 0)` stays `0` for every `k`. The NΔM entry `(0, 0)` after step `k` is
`tanhIter k (-1)` (the `k`-fold composition of `tanh` at `-1`), which is
nonzero by induction using the injectivity of `tanh` and `tanh(0) = 0`.

**Existential form:** `ndm_m2rnn_k_step_separation_exists` says for every
`k ≥ 1` and every fixed-right raw-write resource, there is a `k`-token input
sequence (specifically `kStepWitnessInputs k`) on which the `k`-step
compositions of the resource and NΔM disagree.

**Why this is the natural ceiling reachable from the one-step proof.** The
one-step proof's witness — `lowerLeftState`, `mixedKey`, `v = 0` — was
constructed to expose the cross-row term `−k kᵀ H` that NΔM's delta correction
applies but fixed-right raw writes do not. The k-step extension threads the
same witness through zero-input filler tokens that act as identity on NΔM
(`I − 0·0ᵀ = I`, so the second step reduces to elementwise `tanh`) and as
zero-amplification on M²RNN's row 0 (row locality plus zero-write). The
trajectory gap therefore strictly persists for every finite `k`, ruling out
the "wash-out over composition" failure mode the reviewer worried about.

### Verification

* **Trust gate passes.** `formal/lean/scripts/check_paper_core.sh
  ElmanProofs/PaperCore.lean` reports:
  `trusted check passed: 10 project source files` (was 9 before v16-lean-extend)
  and `paper core check passed: 10 project source files, no native_decide`.
* **No `sorry`, no `axiom`, no `opaque`, no `unsafe`, no `native_decide`** in
  any merged file.
* **No existing theorems renamed.** New theorems live in a new module.
* **`paper/` directory untouched.**

## What did NOT land — the "S₅-coset inseparability with explicit T(d)" target

The brief's Milestone 2 ambition was a theorem of the form:

> For any fixed-weight raw-write RNN with state dimension `d`, there exists a
> constant `T(d)` (with explicit form) such that no such RNN can maintain S₅
> coset state across input sequences of length `T(d)`, while NΔM with the same
> state dimension `d` can.

**This target did NOT land.** The Lean development in v16-lean-extend reaches
"k-step separation on a specific synthetic 2D witness" but not "k-step
inseparability of S₅ coset tracking at general state dimension `d` and
explicit `T(d)`". This section documents why, and what bridging machinery
would be required.

### What blocks the full S₅-coset claim

The S₅-coset inseparability claim is structurally different from the witness
separations in `RecurrentResourceFormalism.lean` and `MultiStepSeparation.lean`.
The witness-based separations exhibit a **specific** input on which NΔM and
M²RNN disagree. The S₅-coset claim is **universal-quantified over inputs from
the S₅ generator alphabet** and requires a **capacity bound**:

1. **Definition of "tracks S₅ coset state".** This is not a witness-based
   property — it is a property of the entire trajectory space. The recurrent
   map `δ : State × InputAlphabet → State` must, after a sequence `g₁ g₂ … g_T`,
   produce a state from which the composition `g_T ∘ … ∘ g₁` (or its coset
   representative) can be decoded. There is no single witness sequence — the
   universal-quantification is over all sequences of length `T`.

2. **A capacity argument on raw-write RNNs.** The negative side
   ("M²RNN-pure with state dim `d` cannot track S₅ past length `T(d)`")
   requires showing that the trajectory space `{δ^T(s₀, g₁, …, g_T) : sequence
   of length T}` does not separate all 120 S₅ permutations — or, more
   strongly, that A₅'s 60 elements collapse together. This is the
   *finite-state-tracking* lower-bound style of result studied by Merrill,
   Petty, Sabharwal ("Illusion of State in State-Space Models") and Liu et al.
   The proof technique is essentially a pigeonhole / circuit-complexity
   argument: the trajectory space is bounded by `f(d, T)` distinguishable
   states, and `f(d, T) < 120` (or `< 60`) for `T < T(d)` is the bound.

3. **No Mathlib analog for the lower bound.** Mathlib has S₅ (`Equiv.Perm
   (Fin 5)`), A₅ simple-group structure
   (`Mathlib.GroupTheory.SpecificGroups.Alternating`), and finite-group
   cardinalities (`Mathlib.Data.Fintype.Card`). What it does **not** have:
   - Formal definitions of NC¹ / TC⁰ circuit classes.
   - Finite-state-tracking lower bounds for parameter-bounded RNNs.
   - Pigeonhole-style capacity arguments specialized to recurrent maps with
     bounded weight matrices.
   - The Merrill et al. / Liu et al. style state-counting arguments.

   Each of these would be a substantial mechanization project. None exists
   off-the-shelf.

4. **The "fixed-weight raw-write RNN with state dim `d`" class is not
   currently typed in Lean.** `RecurrentResourceFormalism.lean` defines
   `FixedRightRawExternalForget2` (and its `KV` generalization) as an
   inductive type with three constructors (row, column, cell), all carrying
   matrix-shaped parameters. To make a capacity statement, the Lean class
   would need to additionally encode:
   - Precision: integer or fixed-precision real weights (capacity bounds
     don't apply to arbitrary-precision reals — Liu et al. use bounded
     precision).
   - A norm / boundedness constraint on `W, r, c, g`.
   - A counting argument over reachable states.

   Each piece is straightforward in isolation but the integration is
   research-grade.

### What partial result we do have, and how it relates

The k-step separation `ndm_m2rnn_k_step_separation` is a **witness** result,
not a **capacity** result. It says: *there exists* a specific 2D input
sequence on which NΔM and any fixed-right raw-write resource disagree, at
every length `k ≥ 1`. It does **not** say that on the S₅ generator alphabet
specifically, M²RNN cannot track cosets at length `T(d)`.

In terms of the reviewer's framing:

> "A one-step advantage could in principle wash out over a trajectory or
> compound. The proof says nothing about which."

The k-step witness separation says **the advantage does compound** (does not
wash out) on a specific synthetic input. It does not yet say the advantage
compounds **on the S₅ generator alphabet** to the specific extent of
distinguishing all 120 S₅ elements at fixed state dim `d`.

### What infrastructure would be needed to push further

The cleanest path to the full S₅-coset inseparability claim in Lean would be:

**Step A. Type the resource class with capacity:**
```
structure BoundedRawWriteRNN (d : ℕ) where
  W : Matrix (Fin d) (Fin d) Real
  Wbound : ‖W‖₂ ≤ B
  -- and similar for r, c, g
```
plus a "precision" or "discretization" axis (Liu et al.-style bounded
precision: each weight is a `k`-bit fixed-point number). Without precision,
the capacity bound fails (an infinite-precision real-weight network can in
principle distinguish unboundedly many trajectories).

**Step B. Define the trajectory map and reachability set:**
```
def reachableStates (R : BoundedRawWriteRNN d) (alphabet : Type)
    (length : ℕ) : Set (Matrix (Fin d) (Fin d) Real) := …
```

**Step C. Prove a pigeonhole / counting lemma:**
```
theorem reachable_count_le (R : BoundedRawWriteRNN d) :
    Fintype.card (reachableStates R alphabet length) ≤ f(d, k_precision, length)
```
where `f` is some explicit (e.g., exponential in `d`, polynomial in length)
upper bound. The Merrill et al. machinery suggests `f(d, k_precision, length) =
poly(d, k_precision)` — independent of length, in the bounded-precision setting.

**Step D. Define "tracks S₅ coset state":**
```
def TracksS5 (system : … → … → State) : Prop :=
  ∀ (w : List S5Tracker.AdjacentGenerator),
    decode (system w) = S5Tracker.run w
```
plus a length parameter `T`.

**Step E. Combine:**
```
theorem ndm_m2rnn_s5_inseparability (d : ℕ) :
    ∃ T : ℕ,
      (∃ ndm_config, TracksS5 (ndm_config.run) d T) ∧
      (∀ R : BoundedRawWriteRNN d, ¬ TracksS5 R.run d T)
```
with `T = poly(d, k_precision) + 1` (or similar).

The hard step is **Step C**. The Mathlib machinery for it doesn't exist; it
would have to be developed from scratch using basic counting / pigeonhole
plus careful arithmetic on the M²RNN update equation.

### Honest assessment of the ceiling

The k-step witness separation is the **clean ceiling reachable from the
existing one-step proof by direct composition**. To push past it to the full
S₅-coset inseparability claim with explicit `T(d)` is **not a translation
exercise**; it is a substantial mechanization project requiring multiple
research-grade lemmas not currently in Mathlib.

The paper's reviewer-acknowledged soft spot — that the rigorous proof covers
only one step while the empirical claim covers full sequences — is not
fully closed by v16-lean-extend. What v16-lean-extend does close:

* **The "wash-out over composition" failure mode is ruled out** for a class of
  fixed-right raw-write resources, on a specific synthetic 2D witness input.
  The gap strictly persists for every finite length.

What remains open:

* The empirical S₅ accuracy gap (NΔM 79% vs M²RNN-CMA 22% at training length,
  widening with length) is not yet linked to a formal capacity bound.
* The S₅ generator alphabet specifically is not yet exercised in the Lean
  separation.
* No explicit `T(d)` bound is in hand.

## Roadmap for v17+ (research-grade, not deadline-driven)

Possible directions, in rough order of decreasing leverage and increasing
mechanization cost:

1. **Complete the KV (general-dimension) k-step separation.** Section
   `KVLift` in `MultiStepSeparation.lean` lands the M2RNN-side row-0
   preservation lemmas at general `K ≥ 2, V ≥ 1`
   (`FixedRightRawExternalForgetKV_preserves_zero_row`,
   `iterUpdateKV_m2rnn_kStep_entry_zero`). What remains is the NDM-side
   `iterUpdateKV_ndm_kStep_entry` lemma — the 2D `fin_cases` discharge does
   not lift directly to `Fin K`; the KV summation needs to be split manually
   into the `l = 0, l = 1, l ≥ 2` cases. This is a tractable but laborious
   piece of explicit matrix arithmetic. Doing it lifts
   `ndm_m2rnn_k_step_separation` to all `K ≥ 2, V ≥ 1`, removing the
   "the 2D witness might be a low-dimensional artifact" objection.

2. **Develop bounded-precision raw-write RNNs.** This is the prerequisite for
   any capacity-based inseparability claim. The work is: define an
   `IsBoundedPrecisionRawWrite` predicate (each weight a `k`-bit fixed-point
   number), wire it through the existing `FixedRightRawExternalForget` class,
   and prove basic closure / arithmetic properties.

3. **Mechanize the Merrill et al. "Illusion of State" counting argument.**
   This is the load-bearing capacity result for finite-state tracking limits
   of bounded-precision RNNs. Without Mathlib pre-existing machinery, this is
   substantial — order of weeks to months of focused Lean work — but it is
   the directly useful artifact for the S₅-coset inseparability target.

4. **Link to the existing S₅ tracker.** Once a capacity bound exists, the
   combination with `S5Tracker.run` and `S5NDMRealization` should be
   relatively short.

5. **Sharpen the FLOP-equivalence and capacity-separation joint claim.**
   The final paper-facing claim would be: at matched per-token FLOP cost and
   matched state dim `d`, NΔM tracks S₅ at length `T(d)` while M²RNN-pure does
   not. The FLOP equivalence is already proved; the capacity separation is
   the remaining gap.

## Files in v16-lean-extend

* `formal/lean/ElmanProofs/Architectures/MultiStepSeparation.lean` — new
  module; imports `M2RNNComparison`, `RecurrentResourceFormalism`,
  `Activations.Lipschitz`. Theorems: `ndm_m2rnn_two_step_separation`,
  `ndm_m2rnn_two_step_separation_exists`, `ndm_m2rnn_k_step_separation`,
  `ndm_m2rnn_k_step_separation_exists`, plus supporting lemmas
  (`tanhIter`, `tanhIter_ne_zero_of_ne_zero`, `tanhIter_tanh_comm`,
  `iterUpdate_zeroSteps_preserves_row0`, `ndm_iter_zero_clean`,
  `ndm_step1_clean_matrix`, `ndm_zero_step_on_clean_state`,
  `FixedRightRawExternalForget2_preserves_zero_row`,
  `m2rnnCandidate_row0_zero_at_zero_row` and per-resource variants).

  The `KVLift` section additionally lands the M2RNN-side row-0 preservation
  lemmas at general `K ≥ 2, V ≥ 1`:
  `m2rnnCandidate_row0_zero_at_zero_row_KV`,
  `m2rnnRowForgetUpdateKV_row0_zero`,
  `m2rnnColumnForgetUpdateKV_row0_zero`,
  `m2rnnCellForgetUpdateKV_row0_zero`,
  `FixedRightRawExternalForgetKV_preserves_zero_row`,
  `iterUpdateKV_zeroSteps_preserves_row0`,
  `iterUpdateKV_m2rnn_kStep_entry_zero`. The NDM-side KV tracking is left
  to v17 (see roadmap item 1).

* `formal/lean/ElmanProofs/PaperCore.lean` — one-line import addition only.

* `formal/lean/LEAN_FRONTIER.md` — this file.

* `formal/lean/scripts/check_paper_core.sh` — **unchanged** (the trust gate is
  load-bearing; tampering with it would defeat the value).

## Concrete commit log

* `lean: two-step separation theorem (ndm_m2rnn_two_step_separation)`
  — Milestone 1 landed.
* `lean: k-step separation theorem (ndm_m2rnn_k_step_separation)`
  — Milestone 2 partial; k-step witness extension; full S₅-coset inseparability
  with explicit T(d) **not** reached; reason documented in this file.
* `lean: document multi-step frontier in LEAN_FRONTIER.md` — this file
  initially landed; subsequently updated to reflect partial KV M2RNN-side
  preservation lemmas (in the same commit cycle as the partial KV section).
