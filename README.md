# When does DoRA help? Geometry and few-shot domain adaptation

This repository contains a small, reproducible research project prepared for the **AIRI Summer School 2026** poster session. It replaces the original single synthetic sanity check with two complementary studies:

1. a controlled **capacity study** that varies adapter rank and row-wise magnitude shift;
2. a trained **real-image proxy benchmark** on `sklearn Digits` under contrast, rotation, and mixed domain shifts.

The question is deliberately narrower than “is DoRA always better than LoRA?”:

> **Under which weight-shift geometry and parameter budget does separating magnitude from direction help?**

## Main findings

- **Positive control:** when the target update is purely additive and rank 4, both rank-4 LoRA and DoRA represent it up to numerical precision.
- **Capacity gap:** at row-wise magnitude strength `γ=0.8`, the best possible additive rank-4 LoRA update has mean relative weight error `0.292`, while a feasible rank-4 DoRA construction has error `1.45e-7`. This is a synthetic representational result, not a trained-model accuracy claim.
- **Real images, fixed rank 4:** DoRA changes test accuracy relative to LoRA by:
  - contrast: `+0.33 pp` (95% paired CI `[-1.51, 2.18]`);
  - rotation: `+0.67 pp` (`[0.20, 1.13]`);
  - mixed shift: `+1.39 pp` (`[0.08, 2.70]`).
- **Negative/control result:** on pure contrast shift, magnitude-only adaptation reaches `97.0%` with only `202` trainable parameters; a full low-rank adapter is unnecessary.
- **Validation-selected rank:** on the mixed shift, DoRA reaches `76.72%`, LoRA `75.28%`, full fine-tuning `73.33%`, and the frozen model `50.83%` (five paired adaptation seeds).

These results support a conditional conclusion: **DoRA is most useful when the target shift contains heterogeneous magnitude changes, but it is not a universal replacement for LoRA.**

## Method in one minute

For a frozen base weight `W₀`, LoRA learns a rank-`r` additive update:

```text
W_LoRA = W₀ + BA
```

DoRA learns the directional update with the same low-rank factors and a separate row-magnitude vector `m`:

```text
V = W₀ + BA
W_DoRA = m ⊙ V / ||V||row
```

The implementation follows the published/PEFT optimization rule and detaches `||V||row` from the backward graph. Both adapters use `α=r`, so the effective scale `α/r` equals one.

### Controlled capacity study

- weight shape: `16 × 32`;
- ground-truth directional update rank: `4`;
- adapter ranks: `1, 2, 4, 8`;
- row-wise magnitude strengths: `0.0, 0.2, 0.4, 0.6, 0.8`;
- ten generated problems per cell;
- LoRA receives its exact truncated-SVD optimum;
- DoRA receives a feasible construction from the known synthetic decomposition.

### Real-image benchmark

- dataset: `sklearn.datasets.load_digits` (`8×8` handwritten digits);
- frozen backbone: MLP `64 → 128 → 64 → 10`, clean test accuracy `97.5%`;
- target training set: 40 transformed samples per class (`400` total);
- shifts: low contrast, rotation, and rotation + contrast + noise;
- methods: frozen, magnitude-only, LoRA, DoRA, full fine-tuning;
- ranks: `1, 2, 4, 8`;
- five paired seeds;
- learning rate and early stopping selected on a held-out target validation split;
- test set is used only after model selection.

## Repository layout

```text
.
├── src/dora_study/          # Adapter, synthetic, and Digits implementations
├── tests/                   # Deterministic correctness checks
├── notebooks/               # Executed analysis companion
├── results/                 # Raw runs, summaries, metadata, and key metrics
├── figures/                 # Poster-ready PNG and SVG figures
├── poster/                  # Final AIRI poster in PPTX and PDF
├── docs/                    # Research report and validation notes
├── run_synthetic.py
├── run_digits.py
├── analyze_results.py
└── make_figures.py
```

## Reproduce

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run_synthetic.py
python run_digits.py
python analyze_results.py
python make_figures.py
python -m unittest discover -s tests -v
```

The complete CPU experiment is intentionally small. Use `--quick` on either experiment script for a smoke test.

## What this project does not claim

- It is **not** a reproduction of the DoRA results on LLaMA, LLaVA, or VL-BART.
- The Digits experiment is a real-data proxy for domain adaptation, not evidence about LLM quality.
- The synthetic DoRA construction uses the known generative decomposition and measures capacity, not trainability.
- Five seeds quantify local variability but are not a substitute for multiple architectures and large downstream benchmarks.
- Wall-clock speed is not compared because it is hardware- and implementation-dependent.

## Sources

1. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9), ICLR 2022.
2. Liu et al., [DoRA: Weight-Decomposed Low-Rank Adaptation](https://proceedings.mlr.press/v235/liu24bn.html), ICML 2024 Oral.
3. [Official NVlabs/DoRA implementation](https://github.com/NVlabs/DoRA).
4. [Hugging Face PEFT LoRA/DoRA documentation](https://huggingface.co/docs/peft/package_reference/lora).
5. Zhang et al., [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://openreview.net/forum?id=lq62uWRJjiY), ICLR 2023.

Author: **Vladislav Lapin**, MIPT FPMI / AI360.
