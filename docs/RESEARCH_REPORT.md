# DoRA under geometric shift: a controlled capacity study and a real-data proxy

## Abstract

Weight-Decomposed Low-Rank Adaptation (DoRA) augments LoRA by learning the magnitude of each weight row separately from a low-rank directional update. The original AIRI proposal tested this mechanism on one synthetic matrix problem and reported a very large error ratio. That result was expected because the target was constructed to belong to the DoRA parameterization, and it did not establish whether the effect persisted across ranks, shift types, or real data.

This study asks a narrower question: **when does magnitude/direction decomposition provide useful adaptation capacity?** It combines (i) a controlled representational sweep with an exact SVD optimum for LoRA and a feasible ground-truth DoRA construction, and (ii) a trained few-shot domain-adaptation benchmark on real handwritten-digit images. The controlled sweep shows no capacity gap in the rank-matched additive positive control, but a rapidly growing gap when row magnitudes become heterogeneous. On the real benchmark, DoRA yields small but positive paired gains over LoRA at rank 4 for rotation and mixed shifts, while a magnitude-only adapter is best for pure contrast. The evidence therefore supports a conditional, not universal, advantage for DoRA.

## 1. Research question and hypotheses

**Question.** Under which target-shift geometry and trainable-parameter budget does DoRA improve upon LoRA?

The preregistered qualitative hypotheses for this small study are:

1. **Additive low-rank control:** when the target is exactly `W₀ + ΔW` and `rank(ΔW) ≤ r`, rank-`r` LoRA should have zero representational residual; DoRA should not receive a structural advantage.
2. **Heterogeneous row magnitudes:** multiplying rows of `W₀ + ΔW` by different factors generally makes the additive update full rank. LoRA should retain an irreducible rank-`r` residual, while DoRA can absorb those factors into its magnitude vector.
3. **Real-data qualification:** the capacity advantage need not translate to a large accuracy gain. When a shift is simple, magnitude-only adaptation may be sufficient; for complex shifts, DoRA may provide a small advantage at low parameter budgets.

## 2. Methods

### 2.1 Implemented adapters

LoRA uses

```text
W = W₀ + (α/r)BA.
```

DoRA uses

```text
V = W₀ + (α/r)BA,
W = m ⊙ V / ||V||row.
```

The normalization term is detached during backpropagation, matching the optimization rule described in the DoRA paper and implemented by Hugging Face PEFT. We set `α=r`, so the effective scale is one for every rank. Matrix `B` is initialized to zero, making both adapters exact no-ops at initialization.

### 2.2 Controlled capacity sweep

For each seed, a base matrix `W₀ ∈ R^(16×32)` and a rank-4 directional shift `ΔW` are sampled. The target is

```text
W★ = diag(1 + γz)(W₀ + ΔW),
```

where `z` is a shuffled row coordinate in `[-1, 1]` and `γ ∈ {0, 0.2, 0.4, 0.6, 0.8}` controls heterogeneous magnitude shift. Adapter ranks are `{1, 2, 4, 8}`, and ten independent problems are generated per cell.

- **LoRA SVD oracle:** the exact best additive rank-`r` update from the truncated SVD of `W★ − W₀`.
- **DoRA construction:** a valid rank-`r` DoRA parameterization using the rank-`r` truncated SVD of the known generative `ΔW` and the target row magnitudes.

The construction is deliberately labeled as a capacity result; it is not a learning benchmark. For `r < 4`, it is feasible but not claimed to be the globally optimal DoRA solution.

Primary metric: relative Frobenius weight error `||W−W★||F / ||W★||F`. Under isotropic Gaussian inputs, the expected output MSE is proportional to the squared Frobenius residual.

### 2.3 Real-image proxy benchmark

The real-data benchmark uses `sklearn.datasets.load_digits`, containing 1,797 images of size `8×8`. A fixed stratified split contains 1,077 source-training images, 360 target-validation images, and 360 target-test images.

A clean MLP (`64→128→64→10`, GELU) is trained once to `97.5%` clean test accuracy. The base model is then frozen and adapted to three target shifts:

- **contrast:** intensity multiplied by `0.10`;
- **rotation:** 25-degree bilinear rotation;
- **mixed:** 18-degree rotation, intensity multiplied by `0.65`, and Gaussian noise `σ=0.16`.

Each adaptation seed receives 40 balanced target examples per class (400 total). LoRA and DoRA are applied to all three linear layers at ranks 1, 2, 4, and 8. Baselines are frozen, magnitude-only, and full fine-tuning. Learning rate is selected from a fixed method-specific grid using target-validation accuracy with validation NLL as the tie-breaker. Early stopping also uses the validation split. The target test split is evaluated only after selection.

Five paired seeds vary the balanced adaptation subset, adapter initialization, minibatch order, and mixed-shift training noise. The pretrained backbone, split, validation corruption, and test corruption are fixed.

## 3. Results

### 3.1 Capacity study

At `γ=0` and rank 4, both LoRA and DoRA reach numerical zero, confirming the additive positive control. At `γ=0.8`, the mean rank-4 LoRA SVD residual grows to `0.2923`, while the feasible DoRA construction remains at `1.45×10⁻⁷`. Rank 8 reduces but does not eliminate the LoRA residual (`0.1579`) because heterogeneous row scaling generally makes the required additive update higher rank.

This result explains the mechanism, but its magnitude must not be read as an empirical accuracy improvement: the target generator explicitly exposes the decomposition DoRA is designed to represent.

### 3.2 Real-data benchmark at fixed rank 4

| Shift | Frozen | Magnitude-only | LoRA r=4 | DoRA r=4 | Full FT |
|---|---:|---:|---:|---:|---:|
| Contrast | 77.50 | **97.00** | 94.72 | 95.06 | 96.39 |
| Rotation | 51.67 | 86.50 | 93.28 | **93.94** | 93.22 |
| Mixed | 50.83 | 65.89 | 73.33 | **74.72** | 73.33 |

Values are mean test accuracy (%) across five paired seeds. The paired DoRA-minus-LoRA differences at rank 4 are:

- contrast: `+0.33 pp`, 95% t-interval `[-1.51, 2.18]`;
- rotation: `+0.67 pp`, `[0.20, 1.13]`;
- mixed: `+1.39 pp`, `[0.08, 2.70]`.

The fixed-rank comparison gives DoRA 2,034 trainable parameters versus 1,832 for LoRA (`+11.0%`). Full fine-tuning updates 17,226 parameters.

### 3.3 Rank behavior and parameter efficiency

DoRA does not dominate at every rank. On rotation, LoRA is slightly better at rank 8 (`94.61%` versus `94.22%`). On mixed shift, DoRA improves over LoRA at every tested rank, with the largest mean differences at rank 1 (`+2.72 pp`) and rank 8 (`+1.89 pp`).

When rank is selected by validation separately for each seed, mixed-shift accuracy is `76.72%` for DoRA, `75.28%` for LoRA, `73.33%` for full fine-tuning, `65.89%` for magnitude-only, and `50.83%` frozen. The apparent advantage over full fine-tuning is consistent with regularization in a 400-example regime, but this small study does not establish a general causal claim.

## 4. Interpretation

The two experiments support the same conditional story:

- DoRA's extra magnitude vector is valuable when row norms need heterogeneous changes that an additive rank-`r` update cannot express cheaply.
- The advantage shrinks when the target can already be represented by a low-rank additive update or when higher rank supplies enough flexibility.
- If the domain shift is almost pure scale, learning only magnitudes can be more parameter-efficient than either LoRA or DoRA.
- Real accuracy gains are much smaller than synthetic residual ratios. A large capacity gap is a mechanism diagnostic, not a promise of an equally large downstream gain.

## 5. Limitations and next experiments

1. `sklearn Digits` is a small real-image proxy, not a transformer or LLM benchmark.
2. One fixed pretrained backbone is reused across five adaptation seeds; uncertainty therefore covers adaptation variability, not pretraining variability or architecture choice.
3. Five seeds give only coarse uncertainty intervals.
4. The synthetic DoRA construction uses the known data-generating decomposition.
5. Clean accuracy with an active target adapter decreases under strong specialization; in deployment the frozen base remains available by disabling the adapter.
6. Training time is not compared because wall-clock values would be hardware- and implementation-dependent.

The highest-value follow-up is a PEFT implementation on a small pretrained transformer with two text tasks, using LoRA, DoRA, and rsLoRA at matched ranks and validation-tuned learning rates. A second follow-up should repeat the image experiment across multiple pretrained backbones and adaptation-set sizes.

## 6. References

1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
2. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral. https://proceedings.mlr.press/v235/liu24bn.html
3. Zhang et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. ICLR 2023. https://openreview.net/forum?id=lq62uWRJjiY
4. Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. 2023. https://arxiv.org/abs/2312.03732
5. Hugging Face PEFT. *LoRA and Weight-Decomposed Low-Rank Adaptation (DoRA).* https://huggingface.co/docs/peft/package_reference/lora
