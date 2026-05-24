# raw/ — Log file pointers

Training logs are large (1–3 MB each, actively growing) and are NOT stored
in the repo. This directory contains only pointer files.

See `../SOURCES.md` for full paths and log-concatenation order.

## Active log locations (on training host)

```
E88/NDM:
  /tmp/pile_convergence_3arch/ctx2k/e88.log              (steps 50–231,000)
  /tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log   (steps 247,550–current, ACTIVE)

FLA-GDN:
  /tmp/pile_convergence_3arch/ctx2k/fla-gdn.log          (steps 50–352,450)
  /tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log   (steps 351,050–current, ACTIVE)

Mamba2:
  /tmp/pile_convergence_3arch/ctx2k/mamba2.log           (steps 50–433,800)
  /tmp/pile_convergence_3arch/ctx2k/mamba2_resume.log    (steps 432,050–current, ACTIVE)

M2RNN-CMA:
  /tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied.log              (steps 50–123,250)
  /tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log   (steps 123,050–current, ACTIVE)

M2RNN-paper (abandoned):
  /tmp/pile_convergence_m2rnn/ctx2k/m2rnn_paper.log      (steps 50–8,400, NOT converged)
```
