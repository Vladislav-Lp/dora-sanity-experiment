[Русская версия](README.md) · **English version**

# When does DoRA help?

**A controlled study of adapter geometry, parameter budgets, and few-shot domain adaptation for AIRI Summer School 2026.**

The Russian version is the primary project page and the destination of the poster QR code. This English page is retained for external readers.

> **Main conclusion:** DoRA produced positive point estimates on new held-out adaptation seeds when the target shift simultaneously required directional change and heterogeneous row-wise rescaling. After multiplicity correction, the evidence remains inconclusive; the result is therefore an internal, carefully bounded estimate rather than universal superiority over LoRA.

## AIRI materials

- [Primary Russian A1 poster, print-ready PDF](poster/Lapin_Vladislav_DoRA_AIRI_2026_RU.pdf)
- [Primary Russian editable PPTX](poster/Lapin_Vladislav_DoRA_AIRI_2026_RU.pptx)
- [English A1 poster, PDF](poster/Lapin_Vladislav_DoRA_AIRI_2026.pdf)
- [English editable PPTX](poster/Lapin_Vladislav_DoRA_AIRI_2026.pptx)
- [Executed analysis notebook](notebooks/AIRI_DoRA_confirmatory_study.ipynb)
- [Technical report](report/DoRA_AIRI_technical_report.html)
- [Full research report](docs/RESEARCH_REPORT.md)
- [Frozen protocol](docs/EXTENSION_PROTOCOL.md)

![Primary Russian poster preview](poster/poster_preview_RU.png)

## Headline result

For the **combined shift** (rotation, contrast change, and noise), DoRA−LoRA was `+1.06 pp` for the MLP and `+0.92 pp` for the CNN. DoRA won 15/20 paired MLP seeds and 8/10 paired CNN seeds. Holm-adjusted p-values were `0.382213` and `0.158955`, so the study does not claim family-wise statistical confirmation or independent external replication.

Negative controls matter: magnitude-only adaptation was best for pure contrast, and parameter-matched LoRA tied DoRA on MLP rotation. The practical hypothesis is therefore geometry-dependent, not “DoRA always wins.”

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_confirmatory.py --architecture mlp
python run_confirmatory.py --architecture cnn
python analyze_extension.py
python run_data_sweep.py
python run_synthetic_optimization.py
python analyze_robustness.py
python make_extension_figures.py --lang en
```

Author: **Vladislav Lapin** · MIPT FPMI / AI360.
