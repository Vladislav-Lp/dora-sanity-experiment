# Post-review claim calibration

Date: 2026-07-17  
Release: `0.3.1`

## Purpose

An external evidence-focused review found no fatal implementation or design error, but identified language that could overstate what the saved experiment establishes. This revision calibrates the claims without rerunning models, changing the frozen protocol, adding post-hoc hypotheses, or hiding negative results.

## What changed

- “replication” and “repeatable advantage” were replaced with “held-seed internal confirmation,” “same-sign estimate,” or “second-backbone check” where appropriate;
- the estimand is explicit: adaptation seeds vary target subsets, corruptions, and optimization while conditioning on one fixed pretrained checkpoint per architecture;
- exact mixed-shift raw and Holm-adjusted paired-t values are shown: MLP `0.063702 → 0.382213`, CNN `0.031791 → 0.158955`;
- the MLP DoRA−LoRA+ adjusted value is shown as `p=0.050935`, explicitly above `0.05`;
- the predeclared budgeted-DoRA comparisons receive a separate secondary-family Holm correction (`m=3`); the mixed result is `+0.82 pp`, CI `[−0.22, +1.86]`, raw `p=0.115696`, adjusted `p=0.347089`, and remains descriptive;
- the LoRA+ `B/A=16` learning-rate ratio is identified as fixed rather than exhaustively tuned;
- Conv2d is described using the implementation's output-filter-as-row convention, and DoRA's exact no-op initialization is stated as `B=0`, `m=||W₀||`;
- notebook provenance is described as in-process execution with embedded outputs and `nbformat` validation.

## What did not change

- no model was retrained;
- no test result affected hyperparameter or rank selection;
- `docs/EXTENSION_PROTOCOL.md` remains the unchanged historical frozen protocol;
- the raw result tables, negative controls, failed synthetic initializations, and limitations remain visible;
- the primary comparison families were not redefined after seeing the results.

## Evidence boundary

The study supports a conditional hypothesis: DoRA may help when a shift combines directional change with heterogeneous magnitude change. It does not establish universal superiority, pretraining-level variability, transformer-scale transfer, or external replication. A separately preregistered multi-checkpoint PEFT study is the appropriate next test.
