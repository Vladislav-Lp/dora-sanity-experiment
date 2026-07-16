from __future__ import annotations

import ast
import base64
import contextlib
import io
import os
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "AIRI_DoRA_geometry_study.ipynb"


def _rich_data(value: object) -> dict[str, object]:
    data: dict[str, object] = {"text/plain": repr(value)}
    html_method = getattr(value, "_repr_html_", None)
    if callable(html_method):
        html = html_method()
        if html:
            data["text/html"] = html
    png_method = getattr(value, "_repr_png_", None)
    if callable(png_method):
        png = png_method()
        if png:
            if isinstance(png, tuple):
                png = png[0]
            if isinstance(png, bytes):
                png = base64.b64encode(png).decode("ascii")
            data["image/png"] = png
    return data


def execute_in_process(notebook: nbf.NotebookNode) -> nbf.NotebookNode:
    """Execute code cells without a Jupyter socket and embed their outputs.

    The hosted workspace blocks ZMQ/TCP kernel sockets. This deterministic
    fallback still validates the exact cells top-to-bottom in one shared Python
    namespace and records standard notebook outputs.
    """
    namespace: dict[str, object] = {"__name__": "__main__"}
    execution_count = 0
    os.chdir(ROOT)
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        source = cell["source"]
        tree = ast.parse(source, filename=f"notebook-cell-{cell_index}", mode="exec")
        final_expression = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        statements = tree.body[:-1] if final_expression is not None else tree.body
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                if statements:
                    module = ast.Module(body=statements, type_ignores=[])
                    exec(compile(module, f"notebook-cell-{cell_index}", "exec"), namespace)
                result = None
                if final_expression is not None:
                    expression = ast.Expression(final_expression.value)
                    result = eval(compile(expression, f"notebook-cell-{cell_index}", "eval"), namespace)
            printed = stdout.getvalue()
            if printed:
                cell["outputs"].append(nbf.v4.new_output("stream", name="stdout", text=printed))
            if result is not None:
                cell["outputs"].append(
                    nbf.v4.new_output(
                        "execute_result",
                        execution_count=execution_count,
                        data=_rich_data(result),
                        metadata={},
                    )
                )
        except Exception as error:
            cell["outputs"].append(
                nbf.v4.new_output(
                    "error",
                    ename=type(error).__name__,
                    evalue=str(error),
                    traceback=[],
                )
            )
            raise
    notebook["metadata"]["execution"] = {
        "status": "validated_top_to_bottom",
        "runner": "in_process_fallback",
        "reason": "Hosted workspace disallows Jupyter kernel sockets.",
    }
    return notebook


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# When does DoRA help?

**A controlled capacity study and a few-shot real-image proxy**  
Vladislav Lapin · MIPT FPMI / AI360 · AIRI Summer School 2026

## tl;dr

- In the additive rank-4 control, LoRA and DoRA both represent the target up to numerical precision.
- Under heterogeneous row-wise magnitude shift (`γ=0.8`), the best rank-4 additive LoRA residual is `0.292`, while a feasible DoRA construction remains at numerical zero.
- On real images at fixed rank 4, DoRA changes accuracy versus LoRA by `+0.33 pp` (contrast), `+0.67 pp` (rotation), and `+1.39 pp` (mixed shift).
- Magnitude-only adaptation is best for pure contrast, so DoRA is useful conditionally rather than universally.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The notebook is an executed companion to the scripts, not the primary training entrypoint. Raw runs are produced by `run_synthetic.py` and `run_digits.py`; this notebook checks their coverage and presents the saved evidence.

### Key assumptions

- The synthetic comparison measures representational capacity: LoRA receives its exact SVD optimum and DoRA receives a feasible construction using the known generative decomposition.
- The real-data benchmark is `sklearn Digits`, not an LLM reproduction.
- Five paired adaptation seeds share one fixed pretrained backbone.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import Image

ROOT = Path.cwd()
assert (ROOT / "results" / "key_metrics.json").exists(), "Run this notebook from the repository root."

key_metrics = json.loads((ROOT / "results" / "key_metrics.json").read_text())
key_metrics"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Controlled capacity sweep"),
        nbf.v4.new_code_cell(
            """synthetic_runs = pd.read_csv(ROOT / "results" / "synthetic" / "synthetic_runs.csv")
assert synthetic_runs["seed"].nunique() == 10
assert set(synthetic_runs["rank"]) == {1, 2, 4, 8}
assert set(synthetic_runs["magnitude_strength"]) == {0.0, 0.2, 0.4, 0.6, 0.8}

capacity = pd.read_csv(ROOT / "results" / "synthetic" / "capacity_gap_summary.csv")
capacity.query("rank == 4")[
    ["magnitude_strength", "lora_relative_error_mean", "dora_relative_error_mean", "capacity_gap_log10_mean"]
].round(6)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "synthetic_capacity_gap.png", width=850, embed=True)"""
        ),
        nbf.v4.new_markdown_cell("### 2. Few-shot Digits domain adaptation"),
        nbf.v4.new_code_cell(
            """digit_runs = pd.read_csv(ROOT / "results" / "digits" / "digits_runs.csv")
adapter_runs = digit_runs[digit_runs["method"].isin(["LoRA", "DoRA"])]
coverage = adapter_runs.groupby(["scenario", "seed", "method"])["rank"].nunique()
assert (coverage == 4).all()
assert digit_runs["target_accuracy"].between(0, 1).all()

rank4 = pd.read_csv(ROOT / "results" / "digits" / "rank4_method_summary.csv")
rank4.pivot(index="scenario", columns="method", values="accuracy_mean_pct").round(2)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "digits_rank4_benchmark.png", width=950, embed=True)"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Paired DoRA − LoRA differences"),
        nbf.v4.new_code_cell(
            """paired = pd.read_csv(ROOT / "results" / "digits" / "paired_delta_summary.csv")
paired.assign(
    interval=paired.apply(lambda row: f"[{row.ci95_low_pp:.2f}, {row.ci95_high_pp:.2f}]", axis=1)
)[["scenario", "rank", "mean_delta_pp", "interval"]].round({"mean_delta_pp": 2})"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "digits_paired_delta.png", width=950, embed=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **The old synthetic result needed a control.** At `γ=0`, rank-4 LoRA is already exact; DoRA's capacity advantage appears only when heterogeneous row scaling makes the additive update higher-rank.
2. **A capacity gap is not an accuracy gap.** Real-data gains are measured in percentage points, not orders of magnitude.
3. **DoRA is most convincing on the mixed shift.** At rank 4 the paired interval is positive, and validation-selected DoRA also leads the other trained methods.
4. **Simpler can be better.** Magnitude-only adaptation wins on pure contrast with 202 parameters.

## Limitations

- small real-image proxy and one fixed backbone;
- five paired adaptation seeds;
- known ground-truth decomposition in the synthetic capacity construction;
- no wall-clock comparison and no LLM/GLUE reproduction.

The next experiment should use the official PEFT implementation on a small pretrained transformer, with LoRA, DoRA, and rsLoRA evaluated under validation-tuned learning rates and matched target modules.
"""
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    executed = execute_in_process(notebook)
    nbf.validate(executed)
    nbf.write(executed, OUTPUT)
    print(f"wrote and executed {OUTPUT}")


if __name__ == "__main__":
    main()
