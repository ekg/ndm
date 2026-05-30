# CMA-ES Log Inventory for v2 Burst Figure

Generated: 2026-05-30T20:24:56Z

This is a read-only inventory of CMA-ES search outputs and related racer
training artifacts visible from this machine. I inspected repository references,
`/home/erikg/elman`, `/home/erikg/emender`, and the obvious `/tmp` racer output
roots referenced by scripts and provenance docs. I did not move, delete, or
modify raw logs or checkpoints.

Machine-readable companion manifest:
`paper/results/cmaes_burst_v2/cmaes_log_manifest.json`.

## Summary

The freshest corrected-parameter-accounting CMA-ES reruns are local and
accessible at:

- `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529`

That path resolves to itself with `readlink -f`; it is not a symlink on this
machine. It targets `1300M` params with `--param_tolerance 0.03`,
`p50k_base`, 2048-token chunks, `/home/erikg/elman/data/pile.txt`, and
15 minutes/eval. Completed corrected reruns now visible there:

| Family | Status at 2026-05-30T20:21:50Z | Best avg loss | Best final loss | Best eval id |
| --- | --- | ---: | ---: | ---: |
| E97 normal | complete | 5.9733408284 | 5.5386 | 59 |
| E97 raw-update | complete | 5.9511015789 | 5.4738 | 58 |
| GDN2 | complete | 6.3849796154 | 5.8386 | 13 |
| E97 linear-state | complete by inspection, though earlier author note called it in-flight | 6.0967407821 | 5.6628 | 34 |
| E88 normal rerun | partial/in-flight, no `results.json` | 6.1086050955 from completed `.done` records | 5.6477 | 27 |
| Transformer rerun | partial/in-flight, no `results.json` | 6.5672682540 from completed `.done` records | 6.5673 | 42 |

The paper-impacting within-class comparison from the corrected runs is visible:
E97 raw-update `5.9511015789` avg loss is ahead of E97 normal
`5.9733408284` avg loss. Both best-eval `params.json` paths and values are
recorded below and in the manifest.

The earlier burst-figure-relevant 2K/1.27B CMA-ES data is also local and
accessible under two roots in `/home/erikg/elman/benchmark_results`:

- `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_warm512_20260526`
- `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526`

These roots use the same stated protocol: `target scale: 1270M`,
`tokenizer: p50k_base`, `context/chunk size: 2048`, schedule-free AdamW,
`candidate time budget: 15 minutes`, `population size: 8`, and
`minimum generations: 8`. The protocol files are present for both roots.

Five 2K/1.27B searches have final `results.json` files:

| Family | Run status | Evaluated configs | `generations.jsonl` rows | Best avg loss | Best final loss | Best eval id |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E88 delta | complete | 128 | 16 | 5.9973678788 | 5.5529 | 99 |
| E88 raw-write | complete | 136 | 17 | 6.0394940476 | 5.5909 | 119 |
| FLA-GDN | complete | 160 | 20 | 6.1103983333 | 5.6165 | 141 |
| M2RNN, `m2rnn_home_xma` | complete | 144 | 18 | 6.0625815217 | 6.0626 | 44 |
| Transformer | complete | 160 | 20 | 5.9045921569 | 5.4683 | 140 |

The same baseline root also contains Mamba2. Mamba2 is accessible and has
per-eval logs, `.done` records, retained checkpoints, and `generations.jsonl`,
but it has no final `results.json`; it should be treated as partial/incomplete.
The best completed Mamba2 eval visible from `.done` records is eval 58 with
avg/final loss `6.0560257485` / `5.6441`.

Emender-side E97/GDN2 CMA-ES attempts are also accessible, but both failed as
usable sweeps. Each has one valid eval and then crashes with
`ValueError: population size 1 is too small with option CMA_mirrors * popsize < 0.5`.
There is no final `results.json`, no `generations.jsonl`, and no root
`protocol.md` or `anchors.json` under the Emender attempt root.

## Fresh Corrected 1.3B Reruns

Root:
`/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529`

Resolved path:
`/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529`

Setup files:

- `anchors_corrected.json`: accessible; used for E97/E97-raw/E97-linear and
  E88 corrected reruns.
- `anchors_gdn2_primary.json`: accessible; used for GDN2.
- `launch_e97_queue.sh`: accessible; launches `e97`, `e97-raw`, and
  `e97-linear` with `--params 1300M`, `--param_tolerance 0.03`,
  `--train_minutes 15`, `--popsize 8`, `--sigma 0.8`, `--chunk_size 2048`,
  `--tokenizer p50k_base`, `--anchor_only_cmaes`, and
  `--use_triton_e88`.
- `launch_gdn2.sh`: accessible; launches `gdn2` with `--params 1300M`,
  `--param_tolerance 0.03`, `--train_minutes 15`, `--popsize 8`,
  `--sigma 0.6`, `--chunk_size 2048`, `--tokenizer p50k_base`,
  `--min_generations 12`, and `--anchor_only_cmaes`.
- Launcher logs under
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/logs`:
  `e97_queue.log`, `gdn2.log`, `e88.log`, `transformer.log`,
  `nohup_test.log`.

Counts and status:

| Family | Run directory | Status | Eval dirs / done / stdout | `generations.jsonl` rows | `results.json` |
| --- | --- | --- | ---: | ---: | --- |
| E97 normal | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97/e97_20260529_152252` | Complete | 64 / 64 / 64 | 8 | yes |
| E97 raw-update | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97-raw/e97-raw_20260530_002342` | Complete | 80 / 80 / 80 | 10 | yes |
| GDN2 | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2/gdn2_20260529_152252` | Complete | 96 / 96 / 96 | 12 | yes |
| E97 linear-state | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97-linear/e97-linear_20260530_113253` | Complete as of inspection | 64 / 64 / 64 | 8 | yes |
| E88 normal rerun | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e88/e88_20260530_151323` | Partial/in-flight at inspection | 36 / 34 / 34 | 7 | no |
| Transformer rerun | `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/transformer/transformer_20260530_042721` | Partial/in-flight at inspection | 60 / 59 / 59 | 7 | no |

Best configs from completed corrected reruns:

- E97 normal, eval 59:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97/e97_20260529_152252/eval_59/params.json`
  contains `dim=2176`, `n_heads=170`, `n_state=32`, `depth=14`,
  `lr=0.0010403731352768883`, `batch_size=2`; `actual_params=1274697304`.
- E97 raw-update, eval 58:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97-raw/e97-raw_20260530_002342/eval_58/params.json`
  contains `dim=2432`, `n_heads=416`, `n_state=16`, `depth=10`,
  `lr=0.0009851067699366818`, `batch_size=3`; `actual_params=1265553184`.
- GDN2, eval 13:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2/gdn2_20260529_152252/eval_13/params.json`
  contains `dim=1664`, `expansion=2`, `depth=10`, `n_heads=58`,
  `lr=0.0020992079360665113`, `batch_size=1`; `actual_params=1262599748`.
- E97 linear-state, eval 34:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e97-linear/e97-linear_20260530_113253/eval_34/params.json`
  contains `dim=1920`, `n_heads=208`, `n_state=32`, `depth=13`,
  `lr=0.0007080609223547477`, `batch_size=3`; `actual_params=1264700224`.

Best completed configs from partial corrected reruns:

- E88 normal rerun, eval 27:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e88/e88_20260530_151323/eval_27/params.json`
  contains `dim=1792`, `n_heads=383`, `n_state=32`, `depth=11`,
  `lr=0.0015330074269705842`, `batch_size=2`; `actual_params=1305634890`.
- Transformer rerun, eval 42:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/transformer/transformer_20260530_042721/eval_42/params.json`
  contains `dim=2048`, `n_heads=15`, `expansion=5`, `depth=17`,
  `lr=0.0004510648520990279`, `batch_size=8`; `actual_params=1306322944`.

Search-level elapsed hours from completed corrected `results.json` files:

- E97 normal: `9.01373437497351` hours.
- E97 raw-update: `11.15292624089453` hours.
- GDN2: `13.874378373755349` hours.
- E97 linear-state: `8.665143682029512` hours.

`generations.jsonl` wallclock coverage:

- E97 normal: first `2026-05-29T16:31:00Z`, last
  `2026-05-30T00:23:42Z`.
- E97 raw-update: first `2026-05-30T01:34:28Z`, last
  `2026-05-30T11:32:53Z`.
- GDN2: first `2026-05-29T16:39:59Z`, last `2026-05-30T05:15:20Z`.
- E97 linear-state: first `2026-05-30T12:44:05Z`, last
  `2026-05-30T20:12:47Z`.
- E88 partial: first `2026-05-30T16:28:33Z`, last
  `2026-05-30T19:57:03Z`.
- Transformer partial: first `2026-05-30T06:40:16Z`, last
  `2026-05-30T19:33:50Z`.

The corrected rerun per-eval layout differs slightly from the older Elman
roots: completed eval dirs include `params.json`, `batch_size.txt`,
`stdout.txt`, optional `stderr.txt`, and run subdirectories with `args.json`
and retained checkpoints. `stdout.txt` includes step-loss lines matching
`^step\\s+[0-9]+\\s+\\| loss` and `FINAL_LOSS_LAST100` markers; CMA fitness is
AvgLoss over the logged training-loss windows, not `FINAL_LOSS_LAST100`.

## Primary 2K CMA-ES Roots

### E88 Delta and Raw-Write

Root:
`/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_warm512_20260526`

Protocol/config files:

- `protocol.md`: accessible.
- `anchors.json`: accessible.
- Top-level logs: `e88.log`, `e88-raw.log`.

| Family | Run directory | Accessible artifacts |
| --- | --- | --- |
| E88 delta | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_warm512_20260526/e88/e88_20260526_030514` | `results.json`, `generations.jsonl`, 128 `eval_*` dirs, 128 `.done`, 128 `params.json`, 128 `stdout.txt`, retained checkpoint files |
| E88 raw-write | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_warm512_20260526/e88-raw/e88-raw_20260526_030514` | `results.json`, `generations.jsonl`, 136 `eval_*` dirs, 136 `.done`, 136 `params.json`, 136 `stdout.txt`, retained checkpoint files |

Best configs from `results.json`:

- E88 delta, eval 99: `dim=2048`, `n_heads=348`, `n_state=32`,
  `depth=10`, `lr=0.0009972765272808343`, `batch_size=2`,
  `actual_params=1148007536`.
- E88 raw-write, eval 119: `dim=1792`, `n_heads=362`, `n_state=32`,
  `depth=11`, `lr=0.0009412965670480728`, `batch_size=2`,
  `actual_params=1149343356`.

Search-level elapsed hours recorded in `results.json`:

- E88 delta: `36.95879488223129` hours.
- E88 raw-write: `39.782313733498256` hours.

`generations.jsonl` wallclock coverage:

- E88 delta: first `2026-05-26T05:33:19Z`, last
  `2026-05-27T16:02:45Z`.
- E88 raw-write: first `2026-05-26T05:33:07Z`, last
  `2026-05-27T18:52:10Z`.

### Baselines Warmed From 512-Token CMA

Root:
`/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526`

Protocol/config files:

- `protocol.md`: accessible.
- `anchors.json`: accessible.
- Top-level logs: `fla-gdn.log`, `m2rnn.log`, `m2rnn_xma.log`,
  `m2rnn_home_xma.log`, `mamba2.log`, `transformer.log`.

| Family | Run directory | Status and accessible artifacts |
| --- | --- | --- |
| FLA-GDN | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/fla-gdn/fla-gdn_20260526_031318` | Complete. `results.json`, `generations.jsonl`, 160 `eval_*` dirs, 160 `.done`, 160 `params.json`, 160 `stdout.txt`, retained checkpoints. |
| M2RNN, initial | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/m2rnn/m2rnn_20260526_031319` | Complete as a failed/invalid attempt. `results.json` exists with `best_loss=inf`; 128 eval dirs and `.done` files, but no `stdout.txt` and no retained checkpoints. |
| M2RNN, `m2rnn_xma` | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/m2rnn_xma/m2rnn_20260526_131718` | Aborted before first successful eval. One `eval_0` dir with `params.json`; no `.done`, `stdout.txt`, `generations.jsonl`, or `results.json`. |
| M2RNN, `m2rnn_home_xma` | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/m2rnn_home_xma/m2rnn_20260526_132728` | Complete usable M2RNN sweep. `results.json`, `generations.jsonl`, 144 `eval_*` dirs, 144 `.done`, 144 `params.json`, 144 `stdout.txt`, retained checkpoints. |
| Mamba2 | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/mamba2/mamba2_20260527_183522` | Partial/incomplete. No `results.json`; `generations.jsonl` exists; 158 `eval_*` dirs, 157 `.done`, 157 `stdout.txt`, 158 `params.json`, retained checkpoints. `eval_157` has params/args but no `.done` or stdout. |
| Transformer | `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/transformer/transformer_20260526_031319` | Complete. `results.json`, `generations.jsonl`, 160 `eval_*` dirs, 160 `.done`, 160 `params.json`, 160 `stdout.txt`, retained checkpoints. |

Best configs from complete usable searches:

- FLA-GDN, eval 141: `dim=3456`, `expansion=2`, `depth=12`,
  `n_heads=38`, `lr=0.000862655652465934`, `batch_size=2`,
  `actual_params=1147668480`.
- M2RNN `m2rnn_home_xma`, eval 44 by average loss: `dim=2304`,
  `n_heads=612`, `n_state=16`, `depth=10`,
  `lr=0.000560674702835787`, `batch_size=5`,
  `actual_params=1144418688`. Note: the best final loss in this run is eval
  125 with final loss `5.6914`; CMA-ES fitness is average loss, so eval 44 is
  the `results.json` best.
- Mamba2 partial best from `.done`, eval 58: `dim=1920`, `d_state=64`,
  `expand=4`, `depth=27`, `lr=0.0014174671284459639`,
  `batch_size=2`, `actual_params=1209400320`.
- Transformer, eval 140 by average loss: `dim=1664`, `n_heads=10`,
  `expansion=6`, `depth=19`, `lr=0.000516391216711691`,
  `batch_size=4`, `actual_params=1157887744`. The best final loss in this run
  is eval 18 with final loss `5.4118`; CMA-ES fitness is average loss, so eval
  140 is the `results.json` best.

Search-level elapsed hours recorded in complete `results.json` files:

- FLA-GDN: `46.05435635017024` hours.
- M2RNN `m2rnn_home_xma`: `44.982980733513834` hours.
- Transformer: `44.32708335810238` hours.

`generations.jsonl` wallclock coverage:

- FLA-GDN: first `2026-05-26T05:55:13Z`, last
  `2026-05-28T01:16:34Z`.
- M2RNN `m2rnn_home_xma`: first `2026-05-26T16:02:35Z`, last
  `2026-05-28T10:26:26Z`.
- Mamba2 partial: first `2026-05-27T21:14:09Z`, last
  `2026-05-29T14:22:12Z`.
- Transformer: first `2026-05-26T05:33:29Z`, last
  `2026-05-27T23:32:56Z`.

## Emender E97/GDN2 CMA-ES Attempts

Root:
`/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528`

Launch commands and intent are documented in
`docs/HANDOFF_E97_GDN2_CMAES_20260528.md`. The root itself does not contain
`protocol.md` or `anchors.json`.

| Family | Run directory | Status |
| --- | --- | --- |
| E97 | `/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/e97/e97_20260528_175019` | Accessible but failed. Top log has 17 generations with 0 valid configs, then one valid eval in generation 18 and a CMA update crash. One `.done`, one `stdout.txt`, one checkpoint, no `generations.jsonl`, no `results.json`. |
| GDN2 | `/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/gdn2/gdn2_20260528_175027` | Accessible but failed. One valid eval in generation 1 and the same CMA update crash. One `.done`, one `stdout.txt`, one checkpoint, no `generations.jsonl`, no `results.json`. |

Best/only evals:

- E97 eval 0: avg/final loss `6.326023684210527` / `6.3260`;
  `dim=3072`, `n_heads=79`, `n_state=32`, `depth=21`,
  `lr=0.0012211070661387916`, searched `batch_size=61`,
  actual `batch_size=10`, `actual_params=1147558806`.
- GDN2 eval 0: avg/final loss `7.157100000000001` / `7.1571`;
  `dim=2688`, `expansion=1`, `depth=20`, `n_heads=32`,
  `lr=0.0009569170744159106`, searched `batch_size=80`,
  actual `batch_size=6`, `actual_params=1357913856`.

These attempts should not be used as complete CMA-ES sweeps.

## Related Racer Logs and Checkpoints

The long racer logs are not CMA-ES sweep logs, but they are the obvious related
artifacts for selected model runs and are referenced by repository scripts and
figure provenance.

Accessible roots:

- `/tmp/pile_convergence_3arch/ctx2k`
- `/tmp/pile_convergence_m2rnn/ctx2k`
- `/tmp/figure2_refresh_snapshot_20260529T180451Z`

Key logs/checkpoint roots referenced by `paper/results/figure_2/SOURCES.md`,
`paper/results/figure_2/AS_OF.md`, and release/audit docs:

- E88/NDM logs: `/tmp/pile_convergence_3arch/ctx2k/e88.log`,
  `/tmp/pile_convergence_3arch/ctx2k/e88_repair_from231k.log`,
  `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log`; checkpoints under
  `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832`.
- FLA-GDN logs: `/tmp/pile_convergence_3arch/ctx2k/fla-gdn.log`,
  `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log`; checkpoints under
  `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832`.
- Mamba2 logs: `/tmp/pile_convergence_3arch/ctx2k/mamba2.log`,
  `/tmp/pile_convergence_3arch/ctx2k/mamba2_resume.log`; checkpoints under
  `/tmp/pile_convergence_3arch/ctx2k/mamba2_resume_ckpt/levelmamba2_1270M_20260511_233832`.
- M2RNN logs: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied.log`,
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log`,
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_paper.log`; checkpoints under
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023`.
- Snapshot copy: `/tmp/figure2_refresh_snapshot_20260529T180451Z` contains
  copied logs and `snapshot_meta.json`.

These racer logs include step and loss trajectories for long training, and the
paper's figure-2 scripts derive BPB/FLOP-normalized curves from them. They do
not contain CMA-ES generation records or per-candidate `.done` files.

## Historical or Diagnostic CMA-ES Archives

These are accessible but should be treated as background or warm-start
evidence, not as the final 2K burst-figure source.

| Root | Families observed | Notes |
| --- | --- | --- |
| `/home/erikg/elman/benchmark_results/cmaes_1270M_20260525` | `e88`, `fla-gdn`, `m2rnn-paper`, `mamba2` | 1.27B short-context/unanchored attempt. `protocol.md` and `provenance.json` exist. The protocol says the run should not be used as final provenance. |
| `/home/erikg/elman/benchmark_results/cmaes_1270M_anchored_20260525` | `e88`, `fla-gdn`, `m2rnn`, `mamba2`, `transformer` | 512-token anchored warm-start evidence. `protocol.md` and `anchors.json` exist. `generations.jsonl` exists for each family, but there are no final `results.json` files. |
| `/home/erikg/elman/benchmark_results/cmaes_1270M_e88_raw_20260525` | `e88-raw` | 512-token E88 raw-write ablation. `protocol.md`, `anchors.json`, one `generations.jsonl`, and eval dirs exist; no final `results.json`. |
| `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_20260526` | `e88`, `e88-raw` | Earlier 2K E88 launch seeded only from the long-racer anchor. Superseded by the `warm512` root above. |
| `/home/erikg/elman/benchmark_results/cmaes_converge` | `mamba2`, `fla-gdn`, `e88`, `e1`, `e75`, `e23`; failed/missing final output for `e42` and `mingru` | Legacy 480M convergence sweep documented in `docs/CMA_FLOP_RATE_FINDING.md` and `paper/results/cma_flop_rate/SOURCES.md`. Candidate budget is described there as ~30 minutes, not the v2 15-minute 2K setup. |
| `/home/erikg/elman/benchmark_results/cmaes_32k` | `fla-gdn`, `mamba2`, `e88`, `e1h` | Long-context 32K archive with final `results.json` files. Not v2 burst source. |
| `/home/erikg/elman/benchmark_results/cmaes_v9`, `/home/erikg/elman/benchmark_results/cmaes_v9_catchup`, `/home/erikg/elman/benchmark_results/cmaes_v9_optimized_kernels` | `e88`, `fla-gdn`, `mamba2`, `transformer`, `mingru`, `minlstm`, `e1`, E88 variants | Older architecture-search archives with final results. Not part of the 2026-05 v2 2K burst campaign. |
| `/home/erikg/elman/cmaes_logs` | MoM E88 log files | Contains `mom_e88_200m_v3.log`, `mom_e88_480m.log`, `mom_e88_search.log`, `mom_e88_search_fixed.log`, `mom_e88_search_v2.log`. Older E88-only logs, not v2 burst source. |

## Field Availability

CMA-ES controller outputs:

- `generations.jsonl` exists for the complete 2K E88/FLA-GDN/M2RNN/Transformer
  sweeps and the partial Mamba2 sweep. Each row includes `wallclock_utc`,
  `ws_idx`, `gen`, `popsize`, generation fitnesses, best loss so far, best
  params so far, `eval_counter`, and CMA-ES state summaries such as `sigma` and
  `mean`.
- `results.json` exists for complete usable 2K E88 delta, E88 raw-write,
  FLA-GDN, M2RNN `m2rnn_home_xma`, and Transformer runs. It includes `model`,
  `best_loss`, `best_params`, `all_results`, `elapsed_hours`, and
  `total_evals`. `all_results` entries include `params`, `actual_params`,
  `loss`, `final_loss`, `eval_id`, `batch_size`, `target_batch_size`, and
  `phase1_loss`.
- `.done` files exist per completed eval and include `params`,
  `actual_params`, `loss`, `final_loss`, actual `batch_size`,
  `target_batch_size`, `eval_id`, `gpu_id`, and `success`.
- `params.json` exists per generated eval and includes `params`, `model_type`,
  and `eval_id`.
- The corrected 1.3B reruns also write `batch_size.txt` per completed eval,
  recording requested/actual batch-size information alongside `.done` and
  `params.json`.

Per-eval training outputs:

- `stdout.txt` files include step counts and loss values in lines like
  `step 10 ... loss 10.8674 ...`. They also include learning-rate/gradient and
  throughput fields in the training script's standard output format.
- Corrected 1.3B rerun stdout files include `FINAL_LOSS_LAST100`. This is
  recorded as `final_loss`, but it is not the CMA-ES objective; CMA fitness is
  AvgLoss over the logged training-loss windows.
- `stdout.txt` step lines do not include wallclock timestamps. Config identity
  comes from the containing `eval_<id>` directory, `params.json`, and `.done`,
  not from each step line.
- The per-eval run `args.json` files include `seed=42`,
  `train_minutes=15.0`, `chunk_size=2048`, `tokenizer=p50k_base`, `level`,
  and actual batch settings. The top-level CMA-ES `results.json` does not
  repeat the seed.
- Raw CMA-ES logs store losses in natural-log cross-entropy units. BPB is not a
  stored CMA-ES field. Downstream code can derive bits/token from loss if
  needed; the long racer figure scripts also compute BPB and FLOP-normalized
  views from racer logs.

Checkpoint availability:

- Complete 2K CMA-ES sweeps retain only a small number of top checkpoint files
  in `eval_*/*/checkpoint_step_*_loss_*.pt`, plus `latest.pt` where produced.
  The harness prunes non-top checkpoints during the run. This inventory records
  paths and counts only; checkpoint contents were not loaded.

## Duration Answer

The relevant 2K and corrected 1.3B CMA-ES candidates are configured as
15-minute runs:

- Both primary protocol files state `candidate time budget: 15 minutes`.
- Top-level CMA-ES logs print `Training time: 15.0 min/config`.
- Per-eval `args.json` files record `train_minutes: 15.0`.
- Corrected 1.3B launch scripts pass `--train_minutes 15`, and their top-level
  logs also print `Training time: 15.0 min/config`.

The accessible artifacts do not prove exact per-eval wallclock start/end times,
because per-eval stdout step lines do not include timestamps and `.done` files
do not include start/end timestamps. The best-supported statement is therefore:
the 2K CMA-ES evals were configured for 15 minutes each, but exact candidate
wallclock windows are not directly logged.

The complete sweeps themselves are not 15-minute runs. Search-level elapsed
times in `results.json` are much longer:

- Corrected E97 normal: 9.01 hours.
- Corrected E97 raw-update: 11.15 hours.
- Corrected GDN2: 13.87 hours.
- Corrected E97 linear-state: 8.67 hours.
- E88 delta: 36.96 hours.
- E88 raw-write: 39.78 hours.
- FLA-GDN: 46.05 hours.
- M2RNN `m2rnn_home_xma`: 44.98 hours.
- Transformer: 44.33 hours.

Step counts within the final sampled evals also differ by architecture and
batching under the same configured 15-minute budget. For example, sampled final
eval stdout files end at roughly step 1390 for E88 delta, 1450 for E88
raw-write, 1610 for FLA-GDN, 770 for M2RNN `m2rnn_home_xma`, 1360 for partial
Mamba2, 3510 for Transformer, 380 for E97, and 630 for GDN2.

Conclusion: accept the 15-minute value as a configured candidate budget, not as
an independently verified per-eval wallclock measurement. Reject any
interpretation that the complete CMA-ES sweeps are ~15 minutes; complete
corrected 1.3B sweeps span roughly 9 to 14 hours, and complete older 2K roots
span roughly 37 to 46 hours.

## Missing or Incomplete References

- Emender E97/GDN2 root is missing `protocol.md` and `anchors.json`; launch
  intent is only in `docs/HANDOFF_E97_GDN2_CMAES_20260528.md` and top-level
  logs.
- Emender E97/GDN2 attempts have no final `results.json` and no
  `generations.jsonl`.
- Mamba2 2K baseline has no final `results.json`; one generated eval
  (`eval_157`) has params/args but no `.done`/stdout, so it should remain
  partial unless a later recovery artifact is found.
- The initial `m2rnn` and `m2rnn_xma` attempts in the baseline root are not
  usable final sweeps; use `m2rnn_home_xma` for the completed M2RNN 2K sweep.
- Historical 512-token/anchored roots are accessible but explicitly diagnostic
  or warm-start evidence, not final 2K tuning.
- Corrected E88 and Transformer reruns under
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529` have
  `generations.jsonl` and completed eval records but no final `results.json` at
  inspection time. Their logs appear active/recent and should be treated as
  partial snapshots.
