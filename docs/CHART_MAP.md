# Extension chart map

| Figure | Analytical question | Family / form | Evidence grain | Supported takeaway | Palette / non-color distinction | Output |
|---|---|---|---|---|---|---|
| Confirmatory DoRA−LoRA | Does the paired effect replicate across shifts and backbones? | Uncertainty / faceted dot-and-interval | Scenario × architecture; 20 MLP or 10 CNN paired seeds | Mixed shift is positive on both backbones; other shifts are less stable | Teal focus, zero line, filled circle vs open square | `figures/extension/confirmatory_dora_minus_lora.*` |
| Mixed strong baselines | Does MLP DoRA survive LoRA+, extra LoRA rank budget, and multiplicity correction? | Uncertainty / dot-and-interval | Three preregistered paired comparisons, 20 seeds | Point estimates favor DoRA, but adjusted evidence is not uniformly decisive | Single teal root; exact values and Holm p labels | `figures/extension/mixed_strong_baselines.*` |
| Data-regime accuracy | Does the result depend on one target-data budget? | Ordered comparison / line with interval | 4 nested budgets × 20 seeds | DoRA remains competitive across 50–400 examples | Teal focus; solid blue LoRA; dashed/open blue LoRA+ | `figures/extension/data_regime_accuracy.*` |
| Data-regime deltas | How does paired advantage vary with target data? | Uncertainty / grouped dot-and-interval | 4 budgets × 20 paired seeds | DoRA−LoRA is positive at all declared budgets; LoRA+ gap is larger | Teal vs blue; circle vs open square; zero line | `figures/extension/data_regime_paired_deltas.*` |
| Synthetic optimization | Is the geometric capacity result trainable from no-op initialization? | Distribution / jittered problem points on log scale | 10 problems; trained value averages 5 initializations | DoRA usually learns the feasible solution; strong magnitude shift exposes optimization failures | Teal trained, open light-teal feasible, blue LoRA; marker shape and fill | `figures/extension/synthetic_optimization.*` |
| Mixed accuracy by backbone | What absolute accuracy and uncertainty do methods achieve? | Comparison / faceted dot-and-interval | Method × backbone confirmatory summary | DoRA approaches full FT with about 2k parameters and improves on LoRA in both backbones | Teal focus; blue adapters; neutral full/magnitude; direct values | `figures/extension/mixed_accuracy_by_backbone.*` |

All charts are generated from saved raw or validated summary CSVs by
`make_extension_figures.py`. Focused axes are explicitly labeled. Absolute bar
charts are avoided where a zero baseline would compress the inferential
comparison; paired effects retain a visible zero reference.
