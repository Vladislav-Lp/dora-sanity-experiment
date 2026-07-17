# Robustness validation report

## Overall assessment: Ready to share

## Data-regime design checks

- 20 paired seeds are present at each of 50, 100, 200, and 400 target examples.
- Target subsets are class-balanced and nested; the same samples and corruption realization are retained when the budget grows.
- Every method reuses the mixed-shift allocation and learning rate selected before this sweep.

| Target examples | DoRA − LoRA (pp) | 95% CI | Holm p | W/T/L |
|---:|---:|---:|---:|---:|
| 50 | +0.46 | [-0.41, +1.33] | 1 | 13/1/6 |
| 100 | +0.53 | [-0.25, +1.30] | 1 | 15/1/4 |
| 200 | +0.92 | [-0.12, +1.95] | 0.7163 | 13/0/7 |
| 400 | +0.69 | [-0.02, +1.41] | 0.5957 | 11/2/7 |

## Synthetic optimization checks

- The inferential unit is the independently generated matrix problem (`n=10`), not each of its five optimizer initializations.
- Feasible DoRA and SVD-LoRA oracle values are invariant across initialization duplicates.
- All declared problems and initializations are retained and all error metrics are finite and non-negative.

| Magnitude strength | Trained DoRA rel. error | Feasible DoRA | LoRA SVD oracle | Convergence <1e-3 |
|---:|---:|---:|---:|---:|
| 0.0 | 2.54e-06 | 1.58e-07 | 1.29e-07 | 100.0% |
| 0.4 | 1.46e-06 | 1.6e-07 | 0.174 | 100.0% |
| 0.8 | 0.00444 | 1.58e-07 | 0.289 | 88.0% |

## Issues found

- No blocking issue detected.

## Required caveats

- The data-budget sweep reuses one fixed MLP backbone and one designed mixed corruption.
- Synthetic targets are deliberately generated from the DoRA family; they diagnose capacity and optimization, not real-task prevalence.
- The robustness analyses are supporting evidence and do not replace the preregistered confirmatory comparisons.
