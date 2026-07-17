# Validation index

Overall status: **ready to share with explicit caveats**.

The current extension supersedes the earlier five-seed validation note. Two independent analysis passes now validate the confirmatory and robustness tracks.

## Machine-checked experiment coverage

- MLP pilot: 510 rows, five validation-only seeds, no target-test columns.
- MLP confirmatory: 480 rows, 20 new paired seeds.
- CNN pilot: 240 rows, five validation-only seeds.
- CNN confirmatory: 180 rows, 10 new paired seeds.
- target-data sweep: 400 trained models, 20 seeds × 4 nested budgets × 5 methods.
- synthetic optimization: 150 runs, 10 problems × 3 magnitude strengths × 5 initializations.
- deterministic correctness suite: 11/11 tests passing.

## Gates applied

- complete seed/scenario/method/budget coverage;
- duplicate-key rejection;
- metric null, finiteness, and range checks;
- exact reconciliation of validation-selected allocations/LRs with confirmatory runs;
- paired comparison coverage;
- parameter-count tests for uniform and matched allocations;
- Linear and Conv2d no-op initialization tests;
- LoRA+ optimizer-group ratio test;
- nested balanced-subset test;
- synthetic oracle invariance across initialization duplicates;
- problem-level synthetic aggregation to prevent pseudoreplication;
- paired t/Wilcoxon calculations and Holm correction from raw rows;
- poster/report figures generated from saved validated tables.

## Validation artifacts

- `docs/EXTENSION_VALIDATION.md`: confirmatory method, coverage, and statistical spot checks.
- `docs/ROBUSTNESS_VALIDATION.md`: data-regime and synthetic optimization checks.
- `results/extension_validation.json`: machine-readable confirmatory status.
- `results/robustness_validation.json`: machine-readable robustness status.
- `docs/CHART_MAP.md`: visual question, grain, encoding, and QA map.

## Required reader-facing caveats

- Digits is a small real-image proxy, not an LLM-scale reproduction.
- One fixed pretrained instance is used per architecture.
- The target split existed in the exploratory phase; the new protocol and seeds were frozen before the extension, but this is not an untouched external replication.
- Multiple-testing correction makes the strongest individual claims less decisive than their unadjusted p-values.
- The synthetic target belongs to the DoRA family by construction.
- Runtime is diagnostic only and should not be presented as a hardware-independent speed comparison.

## Artifact QA

The self-contained technical HTML report passed canonical artifact validation, payload packaging, and structural verification. Browser-level responsive verification is marked `structural_only` because Chromium is not installed in the workspace; the generated report includes the semantic no-script/print fallback.

The AIRI poster passed the presentation overflow test and the inherited-template fidelity check with zero issues. The final PPTX contains no empty structural placeholders. Its PDF export is a single ISO A1 portrait page (`1683.75 × 2383.88 pt`, equivalent to `594 × 841 mm`), uses embedded fonts, and was re-rendered at 120 dpi for full-page visual inspection after export.
