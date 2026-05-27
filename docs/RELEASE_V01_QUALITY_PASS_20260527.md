# v0.1 HF Private-Staging Quality Pass

Date: 2026-05-27

This note records the quality pass for the workgraph release pipeline under
`quality-pass-20260527-hf-release-v01`. The workgraph task descriptions are the
source of truth for execution; this file is an audit artifact for the review.

## Reviewed Pipeline

The intended sequence is:

1. `release-v01-preflight`
2. `release-v01-racer-checkpoint-pin` and `release-v01-emender-repo-dry-run`
3. `release-v01-local-three-model-smoke`
4. `release-v01-docker-local-smoke`
5. `release-v01-private-hf-staging-upload`
6. `release-v01-docker-private-hf-smoke`
7. `release-v01-tag-and-paper-sync`

The dependency chain in workgraph matches that order:

- preflight is downstream of the quality pass
- racer/checkpoint pinning and repo dry-run both depend on preflight
- local smoke depends on both racer/checkpoint pinning and repo dry-run
- Docker local smoke depends on local smoke
- private HF staging depends on Docker local smoke
- Docker private-HF smoke depends on private HF staging
- tag/paper sync depends on Docker private-HF smoke

## Corrections Applied

The release task descriptions were tightened to make the execution guardrails
uniform:

- HF repositories must remain private until explicit user approval is present.
- Passing validation is not, by itself, approval to make repositories public.
- Private staging must not create immutable `v0.1` tags; tagging is reserved for
  the final tag/paper-sync task after private-HF container smoke passes.
- Tasks must not stage or commit tokens, checkpoints, safetensors, HF caches,
  Docker layers, generated PDFs, or other large generated artifacts.
- Validation criteria now ask for concrete evidence such as exact commands,
  revisions, model/config details, auth checks without token disclosure, and
  logged blockers where a resource is unavailable.
- The repo dry-run title was changed from "public checkout" to
  "public-readiness dry run" to avoid implying that a remote should be made
  public during the dry run.

Each downstream release task was also sent a workgraph message instructing the
assigned agent to re-read the updated task description before acting.

## Legacy Note

The older paused task `hf-private-staging-127b-trio-container-smoke` remains
outside the reviewed `release-v01-*` dependency chain. It should not be treated
as the active pipeline path unless a human explicitly chooses to revive that
standalone all-in-one task.
