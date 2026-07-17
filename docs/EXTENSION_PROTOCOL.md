# Confirmatory extension protocol (fixed before new test evaluation)

## Objective

The exploratory AIRI study found a geometry-dependent advantage for DoRA on a
small real-image domain-adaptation proxy. This extension asks whether the result
survives stronger LoRA baselines, nearly matched parameter budgets, new
adaptation seeds, a second architecture, and changes in the amount of target
data.

The extension is intentionally a controlled small-model study. It is not a
claim about transformer-scale performance and is not presented as a reproduction
of the LLaMA/LLaVA experiments in the DoRA paper.

## Frozen decisions

- Dataset and source/validation/test split: the existing stratified
  `sklearn Digits` split with split seed `2026`.
- Target shifts: contrast, rotation, and mixed corruption, with the existing
  fixed validation and test corruptions.
- Primary architecture: MLP `64 -> 128 -> 64 -> 10`.
- Architecture replication: a compact CNN with two convolutional and two
  linear layers.
- Main adapter rank: `r=4`.
- Pilot adaptation seeds: `11, 22, 33, 44, 55`.
- MLP confirmatory seeds: `101..120` (not used in the exploratory study).
- CNN confirmatory seeds: `201..210`.
- Data-regime seeds: `301..320`.
- Optimizer: AdamW, weight decay `1e-4`; maximum `120` epochs; validation
  early stopping with patience `18`.
- Test metrics are evaluated only after validation-based selection.

## Baselines

1. Frozen backbone.
2. Magnitude-only adaptation.
3. LoRA `r=4` on every adapted layer.
4. LoRA+ `r=4`, which uses a larger learning rate for `B` than for `A`.
5. DoRA `r=4` on every adapted layer.
6. Full fine-tuning.
7. **LoRA matched to the DoRA budget.** For the MLP, validation selects one
   allocation from `(5,4,4)`, `(4,5,4)`, and `(4,4,6)`. The closest options
   use 2,024 parameters versus 2,034 for uniform DoRA `r=4`.
8. **DoRA matched to the LoRA budget.** Validation selects one allocation from
   `(4,3,3)`, `(3,4,3)`, and `(3,3,6)`. These use no more trainable parameters
   than uniform LoRA `r=4`.

The allocation order is `(fc1, fc2, classifier)`. A single allocation and
learning rate are selected per shift from pilot validation means and then frozen
for every confirmatory seed.

## Hyperparameter selection

- Standard adapter and magnitude grids: `0.003, 0.01, 0.03`.
- Full fine-tuning grid: `0.0001, 0.0003, 0.001`.
- LoRA+ grid for the `A` learning rate: `0.0003, 0.001, 0.003`; `B/A=16`.
- Selection criterion: highest mean target-validation accuracy across the five
  pilot seeds, with mean validation NLL as the tie-breaker.
- Neither confirmatory accuracy nor confirmatory NLL may change the selected
  configuration.

## Primary comparisons and statistics

The primary paired comparisons are DoRA versus LoRA, LoRA+, and
budget-matched LoRA for each of the three shifts. Secondary comparisons include
budget-matched DoRA versus uniform LoRA.

For each comparison report:

- paired mean test-accuracy difference in percentage points;
- 95% paired Student t interval;
- paired effect size `d_z`;
- paired t-test p-value;
- Wilcoxon signed-rank p-value;
- Holm correction across the family of primary comparisons.

The mixed shift is the main positive hypothesis. Rotation is a secondary
positive hypothesis. Contrast is an explicit negative-control setting in which
magnitude-only adaptation may be sufficient.

## Robustness tracks

### Architecture replication

Repeat the validation-selection and confirmatory procedure on the compact CNN.
This track is reported separately because it changes the backbone and adapter
parameter counts.

### Target-data sweep

On the mixed shift, train with `5, 10, 20, 40` examples per class (50, 100,
200, 400 total) using the MLP confirmatory configuration. Compare DoRA, LoRA,
LoRA+, and both budget-matched variants across the 20 fixed data-regime seeds.
Within each seed, the balanced subsets are nested and a sample retains the same
mixed-corruption realization as the budget grows.

### Synthetic optimization diagnostic

Separate representational capacity from trainability by optimizing DoRA from
its no-op initialization on new synthetic problems at magnitude strengths
`0.0, 0.4, 0.8` and rank `4`. Use problem seeds `200..209` and five adapter
initializations per problem (`0..4`, mapped deterministically to actual RNG
seeds). For each initialization, optimize for at most `2,000` steps and select
the lowest final matrix MSE from learning rates `0.01, 0.03, 0.1`. Compare the
trained result with the feasible DoRA construction and the exact LoRA SVD
oracle. Aggregate inferential summaries first within problem and then across
the ten independent problem seeds, rather than treating five initializations
of one problem as independent problems. This remains a mechanism diagnostic,
not downstream evidence.

## Stopping and reporting rule

All declared runs are retained. Failed or numerically invalid runs are reported,
not silently rerun with a different test-informed setting. Claims must include
negative results, parameter counts, uncertainty, and the single-backbone or
small-dataset limitations that remain after the extension.
