# When does DoRA help? A controlled confirmatory study

## Technical summary

Weight-Decomposed Low-Rank Adaptation (DoRA) separates each weight row's magnitude from a low-rank directional update. The initial AIRI project demonstrated this mechanism on one target explicitly constructed in the DoRA family. The present extension asks a harder and narrower question: **does the decomposition provide a reproducible practical advantage after stronger LoRA baselines, nearly matched parameter budgets, new seeds, a second architecture, and changes in target-data volume?**

The answer is conditional. On the main mixed domain shift, DoRA improves target-test accuracy over ordinary LoRA by `+1.06 percentage points` in the MLP (95% paired CI `[−0.07, +2.18]`, `n=20`) and `+0.92 pp` in the CNN (`[+0.10, +1.73]`, `n=10`). The sign and scale therefore replicate across two backbones. In the MLP, DoRA also exceeds LoRA+ by `+1.43 pp` (`[+0.45, +2.41]`) and a nearly parameter-matched LoRA allocation by `+0.53 pp` (`[−0.29, +1.35]`). However, Holm correction across the nine declared MLP primary comparisons yields adjusted `p=0.382`, `0.051`, and `0.773`, respectively. The correct conclusion is a **consistent practical signal with incomplete family-wise statistical certainty**, not universal superiority.

The counterexamples are equally informative. Magnitude-only adaptation is best on the contrast control with 202 trainable MLP parameters, and parameter-matched LoRA ties DoRA on MLP rotation. The study therefore supports a geometry-dependent adapter choice: magnitude-only for scale-like shifts, LoRA when an additive low-rank update suffices, and DoRA when direction and heterogeneous magnitude must change together.

## Why the original result was not enough

The old project used one small matrix, rank 4, three seeds, and a target generated inside the DoRA parameterization. Its very large DoRA/LoRA error ratio was mathematically expected. It did not answer five questions required for a defensible research claim:

1. Does LoRA recover the positive control when the update is genuinely additive and rank matched?
2. Can DoRA learn the constructed solution from its standard no-op initialization, rather than merely represent it?
3. Does the effect appear on real data with model training and held-out evaluation?
4. Does it survive a stronger optimizer baseline and nearly equal parameter budgets?
5. Is the result stable across random target subsets, data regimes, and architectures?

The extension addresses all five. It preserves the original mechanism study as an explanatory component, not as the headline downstream evidence.

## Research question and hypotheses

**Question.** Under which target-shift geometry and trainable-parameter budget does separating weight magnitude from direction improve low-rank adaptation?

The protocol-frozen hypotheses were:

- **Additive positive control.** If `W★ = W₀ + ΔW` and `rank(ΔW) ≤ r`, rank-`r` LoRA should have no representational residual; DoRA should not receive an artificial advantage.
- **Heterogeneous magnitude mechanism.** Row-wise scaling of `W₀ + ΔW` generally makes the required additive update higher rank. DoRA can encode the scaling in its magnitude vector, while fixed-rank LoRA retains residual error.
- **Mixed-shift practical signal.** DoRA should be most useful when the target shift combines geometric and intensity/noise changes.
- **Contrast negative control.** A nearly pure intensity rescaling may be handled by magnitude-only adaptation.
- **No universal dominance.** Higher rank, a different backbone, or a simple shift may eliminate or reverse DoRA's advantage.

## Adapter specification

For frozen base weight `W₀`, LoRA uses

```text
W = W₀ + (α/r)BA.
```

DoRA uses

```text
V = W₀ + (α/r)BA,
W = m ⊙ V / ||V||row.
```

`m` contains one trainable magnitude per output row/filter. Matrix `B` is initialized to zero, so LoRA and DoRA reproduce the base model before training. The directional norm is detached during backpropagation, following the published and PEFT optimization rule. `α=r`, hence `α/r=1` for every rank.

The implementation supports both `Linear` and `Conv2d`. A convolutional filter is flattened into one row for its low-rank update and magnitude. Unit tests verify exact no-op initialization for both module types, expected parameter counts, LoRA+ learning-rate groups, nested subsets, deterministic data splitting, and synthetic controls.

## Scope, data, and metric definitions

### Dataset and split

The real-data benchmark uses all 1,797 images in `sklearn.datasets.load_digits`. Pixel intensities are scaled to `[0,1]`. A fixed stratified split, seed 2026, contains:

- 1,077 source-training examples;
- 360 target-validation examples;
- 360 target-test examples.

The same labels and split are used for every method. Validation and test corruptions are fixed per scenario. Adaptation subsets and their training corruption vary by paired seed.

### Backbones

- **MLP:** `64 → 128 → 64 → 10`, GELU, three adapted linear layers, 17,226 full-fine-tuning parameters.
- **CNN:** two `3×3` convolutional layers (`1→16→32`), two max-pooling operations, then `128→64→10`, 13,706 full-fine-tuning parameters.

Both fixed pretrained bases reach `97.5%` clean test accuracy.

### Target shifts

- **Contrast:** multiply pixels by `0.10`.
- **Rotation:** bilinear rotation by 25°.
- **Mixed:** 18° rotation, multiply by `0.65`, then Gaussian noise `σ=0.16`.

The mixed shift is the main positive hypothesis; rotation is secondary; contrast is the negative control.

### Metrics

The primary downstream metric is target-test accuracy. Supporting metrics are macro-F1, negative log-likelihood, clean-domain retained accuracy, trainable-parameter count, best validation epoch, and runtime diagnostics. Inferential comparisons operate on within-seed accuracy differences in percentage points.

## Validation-first experimental design

The extension decisions were frozen in `docs/EXTENSION_PROTOCOL.md` before new target-test evaluation.

### Configuration selection

- pilot adaptation seeds: `11, 22, 33, 44, 55`;
- adapter/magnitude learning rates: `0.003, 0.01, 0.03`;
- full-fine-tuning learning rates: `0.0001, 0.0003, 0.001`;
- LoRA+ `A` learning rates: `0.0003, 0.001, 0.003`, with `B/A=16`;
- maximum 120 epochs, AdamW, weight decay `1e−4`, patience 18;
- selection: highest mean target-validation accuracy, then lowest mean validation NLL.

Every method/shift configuration is frozen before confirmatory target-test evaluation. Pilot tables contain no target-test columns, and the validator rejects them if they appear.

### Baselines and parameter controls

1. frozen backbone;
2. magnitude-only;
3. uniform rank-4 LoRA;
4. uniform rank-4 LoRA+;
5. uniform rank-4 DoRA;
6. full fine-tuning;
7. LoRA matched to the DoRA budget;
8. DoRA under the LoRA budget ceiling.

For the MLP, uniform LoRA has 1,832 trainable parameters and uniform DoRA 2,034. Validation selects the matched LoRA allocation from `(5,4,4)`, `(4,5,4)`, `(4,4,6)`; the closest candidates use 2,024 parameters. Budgeted DoRA is selected from `(4,3,3)`, `(3,4,3)`, `(3,3,6)` and never exceeds 1,832 parameters.

### Confirmatory seeds and statistics

- MLP: seeds `101..120`, 20 paired observations per method/shift;
- CNN: seeds `201..210`, 10 paired observations;
- data-regime sweep: seeds `301..320`.

For each declared comparison the analysis reports mean paired difference, 95% paired Student t interval, paired effect size `d_z`, paired t-test, Wilcoxon signed-rank test, wins/ties/losses, and Holm-adjusted p-values. The MLP primary family contains DoRA versus LoRA, LoRA+, and budget-matched LoRA across all three shifts (`m=9`). CNN replication is corrected separately (`m=6`).

## The mixed-shift signal replicates across backbones

### Absolute performance

| Backbone / method | Parameters | Accuracy | 95% CI |
|---|---:|---:|---:|
| MLP DoRA | 2,034 | **75.22%** | [74.32, 76.13] |
| MLP full fine-tuning | 17,226 | 75.17% | [74.38, 75.96] |
| MLP budgeted DoRA | 1,768 | 74.99% | [74.05, 75.93] |
| MLP budget-matched LoRA | 2,024 | 74.69% | [73.84, 75.54] |
| MLP LoRA | 1,832 | 74.17% | [73.10, 75.23] |
| MLP LoRA+ | 1,832 | 73.79% | [72.59, 75.00] |
| CNN full fine-tuning | 13,706 | **80.67%** | [79.89, 81.44] |
| CNN DoRA | 1,990 | 80.53% | [79.93, 81.12] |
| CNN LoRA | 1,868 | 79.61% | [78.90, 80.32] |
| CNN LoRA+ | 1,868 | 79.50% | [78.50, 80.50] |

DoRA reaches essentially the same mean accuracy as full fine-tuning with 11.8% of the MLP parameters and 14.5% of the CNN parameters. This is descriptive parameter efficiency on this benchmark, not a universal compression claim.

### Paired DoRA−LoRA effects

| Backbone | Contrast | Rotation | Mixed |
|---|---:|---:|---:|
| MLP, n=20 | −0.19 [−0.54, +0.15] | +0.42 [−0.29, +1.12] | **+1.06 [−0.07, +2.18]** |
| CNN, n=10 | −0.31 [−1.11, +0.50] | +0.36 [−0.38, +1.10] | **+0.92 [+0.10, +1.73]** |

The mixed effect has similar magnitude in two architectures. MLP favors DoRA in 15/20 seeds; CNN in 8/10. Contrast is slightly negative in both, which is consistent with its role as a negative control.

The MLP ordinary-LoRA comparison has unadjusted `p=0.0637` and Holm-adjusted `p=0.382`. The CNN comparison has unadjusted `p=0.0318` and replication-family Holm `p=0.159`. The replicated direction is scientifically useful, but the corrected evidence remains insufficient for a universal binary claim.

## Stronger baselines narrow the interpretation

On MLP mixed shift:

| Comparison | Mean Δ | 95% CI | `d_z` | Holm paired-t p | W/T/L |
|---|---:|---:|---:|---:|---:|
| DoRA − LoRA | +1.06 | [−0.07, +2.18] | 0.44 | 0.382 | 15/0/5 |
| DoRA − LoRA+ | +1.43 | [+0.45, +2.41] | 0.69 | 0.051 | 14/1/5 |
| DoRA − matched LoRA | +0.53 | [−0.29, +1.35] | 0.30 | 0.773 | 11/1/8 |

The matched-budget result shows that the ordinary DoRA−LoRA point estimate is not explained solely by DoRA's extra magnitude parameters. Yet the wide interval also says that 20 seeds do not locate the matched effect precisely.

LoRA+ is not superior in this setup. That should not be generalized beyond the chosen fixed `B/A=16` ratio: the LoRA+ paper and reference implementation treat the optimal ratio as task dependent. Here it is a strong declared optimizer baseline, not an exhaustive LoRA+ tuning study.

## The effect persists across target-data budgets

The mixed-shift data sweep reuses the already selected MLP configurations. For each seed, 50/100/200/400-example subsets are class-balanced and nested. The same sample retains the same corruption realization as the budget grows.

| Target examples | DoRA | LoRA | LoRA+ | DoRA − LoRA, 95% CI |
|---:|---:|---:|---:|---:|
| 50 | 61.47% | 61.01% | 59.01% | +0.46 [−0.41, +1.33] |
| 100 | 65.82% | 65.29% | 62.99% | +0.53 [−0.25, +1.30] |
| 200 | 70.65% | 69.74% | 68.90% | +0.92 [−0.12, +1.95] |
| 400 | 74.86% | 74.17% | 73.32% | +0.69 [−0.02, +1.41] |

DoRA−LoRA is positive at every declared budget, but every individual LoRA interval includes zero. DoRA's advantage over LoRA+ is larger (`+1.54` to `+2.83 pp`) and its paired-t Holm p-value is below 0.05 at all four budgets. These results support robustness of the practical pattern, not a claim that the effect increases monotonically with more data.

## Simple and negative results clarify the mechanism

### Pure contrast favors magnitude-only adaptation

On MLP contrast, magnitude-only achieves `96.90%` with 202 parameters. DoRA reaches `95.18%`, LoRA `95.38%`, and LoRA+ `95.89%`. The analogous CNN magnitude-only result is `96.97%` with 122 parameters. A full low-rank directional update is unnecessary when the target shift is close to scale adjustment.

### Rotation removes the matched-budget advantage

On MLP rotation, uniform DoRA and selected budget-matched LoRA both reach `93.89%`. DoRA exceeds ordinary LoRA by only `+0.42 pp`, with CI crossing zero. The rank allocation search can therefore close the gap when the shift is predominantly geometric.

### Clean-domain retention reveals specialization

Active target adapters reduce clean-domain accuracy, particularly for rotation. This does not destroy the original model: disabling the adapter restores the frozen base. The result should be read as specialization cost, not catastrophic irreversible forgetting.

## Capacity is clear; optimization is not always trivial

The synthetic target is

```text
W★ = diag(1 + γz)(W₀ + ΔW),
```

where `rank(ΔW)=4` and `γ` controls heterogeneous row scaling. Ten independent problems use seeds `200..209`; each trained DoRA model has five adapter initializations and up to 2,000 optimization steps over learning rates `0.01, 0.03, 0.1`.

| Magnitude strength | Trained DoRA error | Feasible DoRA | LoRA SVD oracle | Convergence `<1e−3` |
|---:|---:|---:|---:|---:|
| 0.0 | 2.54e−6 | 1.58e−7 | 1.29e−7 | 100% |
| 0.4 | 1.46e−6 | 1.60e−7 | 0.174 | 100% |
| 0.8 | 0.00444 | 1.58e−7 | 0.289 | 88% |

At `γ=0`, LoRA and DoRA both satisfy the additive rank-matched positive control. At `γ=0.4`, trained DoRA converges near the feasible construction while LoRA has an irreducible additive residual. At `γ=0.8`, DoRA remains dramatically closer on average, but six of fifty initializations fail the `<1e−3` threshold. Those runs are retained. This separates three statements that the original project conflated:

- DoRA **can represent** the target;
- LoRA's fixed-rank additive family **cannot represent it exactly**;
- standard DoRA optimization **usually but not always finds** the near-exact solution under extreme magnitude shift.

## Validation and data-integrity results

The extension contains 1,960 saved run records:

- 510 MLP pilot rows, including 495 trained candidates;
- 480 MLP confirmatory rows, including 420 trained models;
- 240 CNN pilot rows, including 225 trained candidates;
- 180 CNN confirmatory rows, including 150 trained models;
- 400 trained data-regime models;
- 150 synthetic optimization runs.

Eleven unit tests pass. Independent analysis scripts verify:

- complete seed, scenario, method, and budget coverage;
- no duplicate paired keys;
- no target-test fields in pilot tables;
- exact equality between selected validation configurations and confirmatory configurations;
- finite metrics and valid ranges;
- expected parameter counts;
- oracle invariance across synthetic initialization duplicates;
- problem-level, rather than initialization-level, synthetic aggregation;
- headline numbers recomputed from raw CSV rows.

Both `results/extension_validation.json` and `results/robustness_validation.json` report `ready_to_share` with no blocking issue.

## Limitations and uncertainty

1. **Small proxy, not an LLM reproduction.** Digits is useful for many paired runs and controlled corruption, but does not establish transformer behavior.
2. **One fixed pretrained instance per architecture.** Adaptation seeds cover sampling and optimizer variability, not pretraining/checkpoint variability.
3. **Designed shifts.** The corruptions are controlled mechanisms, not a representative distribution of real deployment drift.
4. **Previously explored target split.** The extension protocol and new seed ranges were frozen before the new runs, but the split existed in the exploratory phase. This is an internal confirmation, not a fully untouched external replication.
5. **Multiplicity reduces certainty.** Several attractive unadjusted results are not below 0.05 after Holm correction.
6. **LoRA+ ratio is fixed.** Only the base learning rate is selected; the `B/A=16` ratio is not tuned per shift.
7. **Synthetic generator favors the mechanism by design.** It diagnoses expressivity and optimization, not real-task frequency.
8. **Wall-clock values are implementation-specific.** Runtime is saved for diagnostics but is not promoted as a cross-method performance claim.

## Recommended next experiment

The highest-value next step is a separately preregistered external replication with the official PEFT stack on a small pretrained transformer. It should compare LoRA, LoRA+, rsLoRA, and DoRA on identical target modules, use matched parameter budgets, tune only on validation, vary at least several pretrained checkpoints, and name one primary task/metric before test evaluation.

The present project should not add an unrun transformer result to the poster as future-looking decoration. Its current contribution is already complete and defensible: a mechanism result, trained optimization diagnostic, real-data confirmation, architecture replication, data-regime robustness, stronger baselines, and explicit uncertainty.

## Conclusion

DoRA is not a universally better LoRA. It is a useful inductive bias when adaptation requires coordinated directional change and heterogeneous weight-magnitude adjustment. In the main mixed shift, the benefit is about one percentage point and repeats across two backbones and four data budgets. In simpler or differently structured shifts, the gain disappears and a cheaper adapter can win. This conditional result is smaller than the original synthetic ratio, but it is far more credible and more useful.

## References

1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
2. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral. https://proceedings.mlr.press/v235/liu24bn.html
3. Hayou et al. *LoRA+: Efficient Low Rank Adaptation of Large Models*. ICML 2024. https://proceedings.mlr.press/v235/hayou24a.html
4. Zhang et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. ICLR 2023. https://openreview.net/forum?id=lq62uWRJjiY
5. Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. 2023. https://arxiv.org/abs/2312.03732
6. Hugging Face PEFT. *LoRA and Weight-Decomposed Low-Rank Adaptation (DoRA).* https://huggingface.co/docs/peft/package_reference/lora
