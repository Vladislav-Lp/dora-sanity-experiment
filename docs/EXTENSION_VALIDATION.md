# Extension validation report

## Overall assessment: Ready to share

## Methodology and data-integrity checks

- Confirmatory rows checked: 660.
- Pilot tables contain validation metrics only; target-test columns are rejected by the validator.
- Every confirmatory row is paired by scenario and seed, and its allocation/LR is reconciled with the frozen selection table.
- Accuracy, macro-F1, NLL, parameter counts, duplicate keys, method coverage, and seed coverage are checked from raw rows.
- Holm correction is applied separately to the predeclared MLP primary (`m=9`), CNN (`m=6`), and descriptive MLP secondary (`m=3`) families.

## Issues found

- No blocking calculation or coverage issue detected.

## Primary mixed-shift spot checks

| Comparison | Mean Δ (pp) | 95% CI | Holm p (paired t) | W/T/L |
|---|---:|---:|---:|---:|
| DoRA − lora | +1.06 | [-0.07, +2.18] | 0.3822 | 15/0/5 |
| DoRA − lora_plus | +1.43 | [+0.45, +2.41] | 0.05094 | 14/1/5 |
| DoRA − lora_matched | +0.53 | [-0.29, +1.35] | 0.7728 | 11/1/8 |

## Descriptive secondary-family spot check

The budgeted-DoRA comparison was declared secondary and is not used to rescue the primary claim.

| Comparison | Mean Δ (pp) | 95% CI | Raw p | Secondary-family Holm p | W/T/L |
|---|---:|---:|---:|---:|---:|
| budgeted DoRA − LoRA (mixed) | +0.82 | [-0.22, +1.86] | 0.1157 | 0.3471 | 15/0/5 |

## Required caveats

- Digits is a small vision proxy, not an LLM-scale reproduction.
- The extension uses one fixed pretrained backbone per architecture; adaptation seeds quantify few-shot sampling and optimization variability, not pretraining variability.
- The three corruption scenarios are designed domain shifts; conclusions should be framed as geometry-dependent rather than universal superiority.
- The same target split appeared in the earlier exploratory phase. Hyperparameters and new seed ranges were frozen before the extension, but this is a protocol-frozen internal confirmation rather than a fully untouched external replication.

## Reproducible evidence

- `results/extension_accuracy_summary.csv`
- `results/extension_paired_comparisons.csv`
- `results/confirmatory_mlp/selected_configs.csv`
- `results/confirmatory_cnn/selected_configs.csv`
