# Validation index

Overall status: **ready to share with explicit caveats**.

The current extension supersedes the earlier five-seed validation note. Two separate analysis passes validate the held-seed and robustness tracks.

## Machine-checked experiment coverage

- MLP pilot: 510 rows, five validation-only seeds, no target-test columns.
- MLP held-seed evaluation: 480 rows, 20 new paired adaptation seeds.
- CNN pilot: 240 rows, five validation-only seeds.
- CNN held-seed evaluation: 180 rows, 10 new paired adaptation seeds.
- target-data sweep: 400 trained models, 20 seeds × 4 nested budgets × 5 methods.
- synthetic optimization: 150 runs, 10 problems × 3 magnitude strengths × 5 initializations.
- deterministic correctness suite: 11/11 tests passing.

## Gates applied

- complete seed/scenario/method/budget coverage;
- duplicate-key rejection;
- metric null, finiteness, and range checks;
- exact reconciliation of validation-selected allocations/LRs with held-seed runs;
- paired comparison coverage;
- parameter-count tests for uniform and matched allocations;
- Linear and Conv2d no-op initialization tests;
- LoRA+ optimizer-group ratio test;
- nested balanced-subset test;
- synthetic oracle invariance across initialization duplicates;
- problem-level synthetic aggregation to prevent pseudoreplication;
- paired t/Wilcoxon calculations and Holm correction from raw rows, separately for MLP primary (`m=9`), CNN (`m=6`), and descriptive MLP secondary (`m=3`) families;
- poster/report figures generated from saved validated tables.

## Validation artifacts

- `docs/EXTENSION_VALIDATION.md`: held-seed method, coverage, and statistical spot checks.
- `docs/ROBUSTNESS_VALIDATION.md`: data-regime and synthetic optimization checks.
- `results/extension_validation.json`: machine-readable held-seed status.
- `results/robustness_validation.json`: machine-readable robustness status.
- `docs/CHART_MAP.md`: visual question, grain, encoding, and QA map.

## Required reader-facing caveats

- Digits is a small real-image proxy, not an LLM-scale reproduction.
- One fixed pretrained instance is used per architecture.
- The target split existed in the exploratory phase; the new protocol and seeds were frozen before the extension, but this is not an untouched external replication.
- Adaptation seeds are conditional on one fixed pretrained checkpoint per architecture; they do not measure pretraining variability.
- Multiple-testing correction makes the strongest individual claims less decisive than their unadjusted p-values.
- The LoRA+ `B/A=16` learning-rate ratio is fixed rather than exhaustively tuned per shift.
- The synthetic target belongs to the DoRA family by construction.
- Runtime is diagnostic only and should not be presented as a hardware-independent speed comparison.

## Artifact QA

The self-contained technical HTML report passed canonical artifact validation, payload packaging, and structural verification. Browser-level responsive verification is marked `structural_only` because Chromium is not installed in the workspace; the generated report includes the semantic no-script/print fallback.

The notebook is generated and executed in-process by `scripts/build_notebook.py`, validated with `nbformat`, and saved with embedded outputs and explicit execution provenance. It should not be described as an untouched external kernel run.

The AIRI poster passed the presentation overflow test and the inherited-template fidelity check with zero issues. The final PPTX contains no empty structural placeholders. Its PDF export is a single ISO A1 portrait page (`1683.75 × 2383.88 pt`, equivalent to `594 × 841 mm`), uses embedded fonts, and was re-rendered at 120 dpi for full-page visual inspection after export.
