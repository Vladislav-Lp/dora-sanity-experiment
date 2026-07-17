from __future__ import annotations

import ast
import base64
import contextlib
import io
import os
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "AIRI_DoRA_confirmatory_study.ipynb"


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
    """Validate cells top-to-bottom when hosted Jupyter sockets are blocked."""

    namespace: dict[str, object] = {"__name__": "__main__"}
    execution_count = 0
    os.chdir(ROOT)
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        tree = ast.parse(cell["source"], filename=f"notebook-cell-{cell_index}", mode="exec")
        final_expression = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        statements = tree.body[:-1] if final_expression is not None else tree.body
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                if statements:
                    exec(
                        compile(ast.Module(body=statements, type_ignores=[]), f"notebook-cell-{cell_index}", "exec"),
                        namespace,
                    )
                result = None
                if final_expression is not None:
                    result = eval(
                        compile(ast.Expression(final_expression.value), f"notebook-cell-{cell_index}", "eval"),
                        namespace,
                    )
            if stdout.getvalue():
                cell["outputs"].append(nbf.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
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

**A controlled held-seed study of adapter geometry, parameter budgets, and few-shot domain adaptation**  
Vladislav Lapin · MIPT FPMI / AI360 · AIRI Summer School 2026

## Technical summary

- Held-seed DoRA−LoRA estimates are **+1.06 pp** in the MLP and **+0.92 pp** in the CNN; Holm-adjusted paired-t `p=0.382213` and `0.158955` remain inconclusive.
- The same sign appears on two fixed backbones and all four target-data budgets, conditional on one pretrained checkpoint per architecture.
- Magnitude-only wins the pure-contrast control; parameter-matched LoRA ties DoRA on MLP rotation.
- Trained synthetic DoRA usually realizes its capacity advantage, but convergence falls to 88% under the strongest magnitude shift.

The contribution is a conditional result with uncertainty, not a claim that DoRA always beats LoRA.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Protocol and evidence boundary

Pilot seeds select learning rates and MLP rank allocations using target validation only. Held-seed adaptation seeds are disjoint, and target test is evaluated after every configuration is frozen. They quantify subset/corruption/optimization variability conditional on one fixed pretrained checkpoint per architecture. The target split existed in the exploratory phase, so this is a protocol-frozen internal confirmation, not an untouched external replication. The real-data benchmark uses `sklearn Digits`; it is not an LLM reproduction. The exact historical protocol is saved unchanged in `docs/EXTENSION_PROTOCOL.md`.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image

ROOT = Path.cwd()
extension_audit = json.loads((ROOT / "results" / "extension_validation.json").read_text())
robustness_audit = json.loads((ROOT / "results" / "robustness_validation.json").read_text())
assert extension_audit["status"] == "ready_to_share" and not extension_audit["issues"]
assert robustness_audit["status"] == "ready_to_share" and not robustness_audit["issues"]
{"confirmatory": extension_audit, "robustness": robustness_audit}"""
        ),
        nbf.v4.new_markdown_cell("## Validation-selected configurations"),
        nbf.v4.new_code_cell(
            """mlp_selected = pd.read_csv(ROOT / "results" / "confirmatory_mlp" / "selected_configs.csv", dtype={"allocation": str})
cnn_selected = pd.read_csv(ROOT / "results" / "confirmatory_cnn" / "selected_configs.csv", dtype={"allocation": str})

selected = pd.concat([mlp_selected, cnn_selected], ignore_index=True)
selected[[
    "architecture", "scenario", "method_id", "allocation", "learning_rate",
    "trainable_parameters", "validation_accuracy_mean"
]].sort_values(["architecture", "scenario", "method_id"]).round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Held-seed result: mixed-shift estimates are positive on both backbones

Intervals below are paired within architecture and adaptation seed. Similar point estimates in the MLP and CNN are useful internal evidence, while the intervals and adjusted p-values prevent describing this as family-wise confirmation or cross-checkpoint replication.
"""
        ),
        nbf.v4.new_code_cell(
            """comparisons = pd.read_csv(ROOT / "results" / "extension_paired_comparisons.csv")
comparisons.query("comparison == 'dora_vs_lora'")[[
    "architecture", "scenario", "mean_delta_pp", "ci95_low_pp", "ci95_high_pp",
    "paired_effect_dz", "paired_t_p", "paired_t_p_holm", "wins", "ties", "losses", "n_pairs"
]].round(4)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "extension" / "confirmatory_dora_minus_lora.png", width=950, embed=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Stronger and nearly parameter-matched baselines

The fixed LoRA+ `B/A=16` ratio is a declared optimizer baseline, not an exhaustive LoRA+ tuning study. Its mixed-shift Holm-adjusted value is `p=0.050935`, above `0.05`. Budgeted DoRA is a separate descriptive secondary comparison (`+0.82 pp`, CI `[−0.22, +1.86]`, raw `p=0.115696`, secondary-family Holm `p=0.347089`) and is not used to rescue the primary claim.
"""
        ),
        nbf.v4.new_code_cell(
            """mixed_primary = comparisons.query(
    "architecture == 'mlp' and scenario == 'mixed' and numerator == 'dora'"
)[[
    "comparator", "mean_delta_pp", "ci95_low_pp", "ci95_high_pp", "paired_effect_dz",
    "paired_t_p_holm", "wilcoxon_p_holm", "wins", "ties", "losses"
]]
mixed_primary.round(4)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "extension" / "mixed_strong_baselines.png", width=850, embed=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Target-data sweep

The 50/100/200/400-example subsets are class-balanced and nested. A retained sample keeps the same corruption realization, and every method reuses the mixed-shift configuration selected before this sweep.
"""
        ),
        nbf.v4.new_code_cell(
            """data_accuracy = pd.read_csv(ROOT / "results" / "data_sweep_mlp" / "data_sweep_accuracy_ci.csv")
data_delta = pd.read_csv(ROOT / "results" / "data_sweep_mlp" / "data_sweep_paired_comparisons.csv")
data_accuracy[data_accuracy["method_id"].isin(["dora", "lora", "lora_plus"])][[
    "adaptation_examples", "method", "accuracy_mean_pct", "accuracy_ci95_low_pct", "accuracy_ci95_high_pct"
]].round(2)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "extension" / "data_regime_accuracy.png", width=850, embed=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Synthetic capacity versus trainability

LoRA receives its exact rank-4 SVD optimum. Feasible DoRA uses the known generating decomposition. Trained DoRA starts from the standard no-op initialization; the inferential unit is the independent matrix problem, not each optimizer initialization.
"""
        ),
        nbf.v4.new_code_cell(
            """synthetic = pd.read_csv(
    ROOT / "results" / "synthetic_optimization" / "synthetic_optimization_verified_summary.csv"
)
synthetic.round(7)"""
        ),
        nbf.v4.new_code_cell(
            """Image(filename=ROOT / "figures" / "extension" / "synthetic_optimization.png", width=850, embed=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Interpretation and limits

1. **Geometry is a testable hypothesis.** Mixed direction+magnitude change is the setting in which DoRA has the clearest positive conditional point estimate here.
2. **Parameter count is not the whole explanation.** DoRA remains ahead of a LoRA allocation within ten parameters of its MLP budget, although the interval is wide.
3. **Capacity is not optimization.** The synthetic target is exactly representable, yet strong magnitude shift creates failed DoRA initializations.
4. **Simpler can be better.** Magnitude-only adaptation wins on contrast, and matched LoRA closes rotation.
5. **External validity remains open.** One fixed Digits checkpoint per architecture is neither cross-checkpoint replication nor transformer-scale evidence.

The next high-value study is an external, separately frozen PEFT experiment on a small pretrained transformer with matched target modules and multiple base checkpoints.
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
