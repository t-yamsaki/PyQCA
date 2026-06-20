# PyQCA

**PyQCA** is a Python-native toolkit for **Qualitative Comparative Analysis (QCA)**.

[日本語 README](README.ja.md)

It provides a unified framework for running **crisp-set QCA (csQCA)**, **multi-value QCA (mvQCA)**, **fuzzy-set QCA (fsQCA)**, and mixed-type configurational analysis in Python. PyQCA also supports pluggable minimization backends, including standard Boolean minimization and set-covering-based minimization, as well as threshold-sweep analysis inspired by the ThS-QCA framework.

> Status: experimental / under active development

---

## Why PyQCA?

QCA is widely used in social science, management research, political science, evaluation research, and configurational causal analysis. Mature QCA tooling exists in R, especially for crisp-set, multi-value, and fuzzy-set QCA workflows.

PyQCA aims to provide a Python-native QCA foundation for researchers and practitioners who want to:

* run csQCA, mvQCA, and fsQCA directly in Python;
* analyze crisp, multi-valued, and fuzzy-set conditions in a unified interface;
* switch between different minimization algorithms;
* inspect consistency, coverage, truth tables, and solution formulas;
* conduct threshold-sensitivity analysis systematically;
* integrate QCA workflows with Python data science tools such as pandas, NumPy, scikit-learn, Jupyter, and visualization libraries.

PyQCA is not intended to be only a thin wrapper around existing R packages. The goal is to provide a Python-native architecture for standard QCA workflows and future methodological extensions.

---

## Core Features

### 1. csQCA

Run crisp-set QCA on binary set-membership scores.

```python
from qca import CSQCA

model = CSQCA(
    data=df,
    outcome="Y",
    conditions=["A", "B", "C"],
)

result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
)

print(result.truth_table)
print(result.solutions)
```

---

### 2. mvQCA

Run multi-value QCA on categorical and ordinal conditions.

```python
from qca import MVQCA

model = MVQCA(
    data=df,
    outcome="Y",
    conditions=["A", "B", "C"],
)

result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
)

print(result.solutions)
print(result.consistency)
print(result.coverage)
```

---

### 3. fsQCA

Run fuzzy-set QCA with calibrated set-membership scores.

```python
from qca import FSQCA, calibrate_crisp, calibrate_piecewise

calibrated = calibrate_piecewise(df, "raw_score", out_col="A").df
calibrated = calibrate_crisp(calibrated, "raw_flag", threshold=1, out_col="B").df

model = FSQCA(
    data=calibrated,
    outcome="Y",
    conditions=["A", "B"],
    condition_types={"A": "fuzzy", "B": "crisp"},
)

result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
)

print(result.truth_table)
print(result.solutions)
```

---

### 4. MixedQCA

`MixedQCA` is the high-level interface for datasets containing multiple condition types.

It allows crisp, multi-valued, and fuzzy-set conditions to be analyzed in one workflow.

```python
from qca import MixedQCA

model = MixedQCA(
    data=df,
    outcome="Y",
    conditions=["A", "B", "C", "D"],
    condition_types={
        "A": "crisp",
        "B": "multi",
        "C": "fuzzy",
        "D": "fuzzy",
    },
)

result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
    minimizer="standard",
)

print(result.summary())
```

The same workflow can be built from the v0.2 unified schema:

```python
schema = [
    {"name": "A", "type": "crisp", "domain": [0, 1], "calibrated": True},
    {"name": "B", "type": "multi-value", "domain": ["low", "high"]},
    {"name": "C", "type": "fuzzy", "domain": [0, 1], "calibrated": True},
]

model = MixedQCA.from_schema(data=df, outcome="Y", schema=schema)

print(model.workflow)          # mixed-crisp-fuzzy-multi
print(model.condition_schema)  # normalized schema DataFrame
```

The purpose of `MixedQCA` is to make mixed-type configurational analysis explicit, rather than forcing all variables into a single representation too early.

---

### 5. Pluggable Minimization Backends

PyQCA separates the QCA engine from the minimization algorithm.

This makes it possible to compare standard Boolean minimization with alternative optimization-based approaches.

```python
result = model.fit(
    minimizer="standard",
)
```

```python
result = model.fit(
    minimizer="set_cover",
)
```

Available minimizers:

| Minimizer          | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `standard`         | Standard Boolean minimization backend                |
| `qmc`              | Quine-McCluskey-style minimization                   |
| `set_cover`        | Set-covering-based minimization                      |
| `greedy_set_cover` | Greedy approximation for large candidate rule spaces |
| `exact_set_cover`  | Exact set-covering optimization backend              |

Set-covering-based minimization is designed as an alternative backend, especially useful for large mvQCA-style candidate rule spaces.

Backends can also be benchmarked against the same model and thresholds:

```python
from qca import benchmark_minimizers

bench = benchmark_minimizers(
    model,
    minimizers=["standard", "qmc", "greedy_set_cover", "exact_set_cover"],
    outcome_threshold=0.75,
    consistency_cutoff=0.75,
)

print(bench.summary())
```

---

### 6. ThresholdSweep

PyQCA includes a threshold-sweep analysis layer inspired by the ThS-QCA framework.

Instead of treating thresholds as fixed preprocessing choices, `ThresholdSweep` treats them as explicit analytical variables.

```python
import pandas as pd

from qca import ThresholdSweep

raw_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "X1": [9, 8, 8, 7, 6, 5, 4, 3],
        "X2": [8, 8, 7, 6, 7, 5, 4, 2],
        "X3": [9, 7, 8, 6, 5, 6, 3, 2],
        "Y": [10, 9, 9, 8, 7, 6, 4, 2],
    }
)

sweep = ThresholdSweep(
    raw_df,
    outcome="Y",
    conditions=["X1", "X2", "X3"],
    case_id="case",
)

result = sweep.outcome(
    thresholds=[6, 7, 8, 9],
    condition_thresholds={"X1": 7, "X2": 7, "X3": 7},
    incl_cut=0.8,
    n_cut=1,
    coverage_cutoff=0.1,
)

print(result.summary_df)
```

`ThresholdSweep` follows the ThSQCA threshold algorithm:

* raw variables are binarized with `x >= threshold`;
* the outcome is always threshold-binarized for each run;
* pre-calibrated condition variables can be passed through with `pre_calibrated`;
* `outcome`, `condition`, `multi_condition`, and `dual` mirror ThSQCA's `otSweep`, `ctSweepS`, `ctSweepM`, and `dtSweep` modes.

Available sweep modes:

| API                       | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| `sweep.outcome()`         | Sweep outcome thresholds                       |
| `sweep.condition()`       | Sweep a single condition threshold             |
| `sweep.multi_condition()` | Sweep a grid of multiple condition thresholds  |
| `sweep.dual()`            | Sweep outcome and condition thresholds jointly |
| `sweep.fuzzy_anchors()`   | Sweep fsQCA calibration anchors                |

---

## Example: Outcome Threshold Sweep

```python
import pandas as pd
from qca.sweep import ThresholdSweep

sweep_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 13)],
        "A": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2],
        "B": [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 2],
        "C": [9, 7, 8, 6, 7, 5, 6, 4, 5, 3, 4, 2],
        "Y": [10, 9, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2],
    }
)

sweep = ThresholdSweep(
    sweep_df,
    outcome="Y",
    conditions=["A", "B", "C"],
    case_id="case",
)

res = sweep.outcome(
    thresholds=[5, 6, 7, 8, 9],
    condition_thresholds={"A": 7, "B": 7, "C": 7},
    minimizer="set_cover",
    incl_cut=0.85,
    n_cut=1,
    coverage_cutoff=0.1,
)

res.summary_df
```

Example output:

| thrY | expression          | consistency | coverage | n_solutions | solution_type | valid |
| ---: | ------------------- | ----------: | -------: | ----------: | ------------- | :---- |
|  5.0 | (A * C) + (A * B)  |        1.00 |     0.56 |           1 | complex       | True  |
|  6.0 | (A * C) + (A * B)  |        1.00 |     0.63 |           1 | complex       | True  |
|  7.0 | (A * C) + (A * B)  |        1.00 |     0.71 |           1 | complex       | True  |
|  8.0 | (A * C) + (A * B)  |        1.00 |     1.00 |           1 | complex       | True  |
|  9.0 | (A * B * C)        |        1.00 |     1.00 |           1 | complex       | True  |

This helps researchers inspect how sufficient configurations emerge, disappear, branch, or become more complex as thresholds change.

---

## Example: fsQCA Calibration Anchor Sweep

For fuzzy-set QCA, threshold sensitivity often appears through calibration anchors: full non-membership, crossover, and full membership.

PyQCA supports anchor sweep analysis through `AnchorSensitivity` and the convenience `ThresholdSweep.fuzzy_anchors()` adapter.

```python
import pandas as pd
from qca import ThresholdSweep

anchor_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 11)],
        "Budget": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Capacity": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        "Y": [0.05, 0.10, 0.20, 0.40, 0.55, 0.50, 0.70, 0.80, 0.90, 0.95],
    }
)

anchor_sweep = ThresholdSweep(
    anchor_df,
    outcome="Y",
    conditions=["Budget", "Capacity"],
    case_id="case",
)

res = anchor_sweep.fuzzy_anchors(
    condition="Budget",
    anchor_grid={
        "full_out": [1, 2, 3],
        "crossover": [5, 6],
        "full_in": [8, 9, 10],
    },
    incl_cut=0.8,
    outcome_threshold=0.75,
)

print(res.stability)
print(res.to_markdown())
res.plot_heatmap()
```

The dedicated API can also sweep one anchor family at a time:

```python
from qca import AnchorSensitivity

anchors = AnchorSensitivity(
    anchor_df,
    outcome="Y",
    conditions=["Budget", "Capacity"],
    case_id="case",
)

res = anchors.crossover(
    "Budget",
    values=[5, 6, 7],
    full_out=1,
    full_in=10,
    incl_cut=0.8,
)
```

This extends threshold-sweep analysis beyond crisp-set dichotomization and into fuzzy-set calibration sensitivity.

---

## Example: Reporting and Reproducibility

```python
from pathlib import Path

from qca import (
    ExperimentLogger,
    collect_reproducibility_metadata,
    generate_markdown_report,
    jupyter_summary,
    to_latex_table,
)
from qca.viz import plot_configuration_chart

result = model.fit(consistency_cutoff=0.8)

reports = Path("reports")
reports.mkdir(parents=True, exist_ok=True)

generate_markdown_report(result, path="reports/qca_result.md")
to_latex_table(result, table="solutions", path="reports/solutions.tex")

metadata = collect_reproducibility_metadata(result, extra={"dataset": "study-1"})
ExperimentLogger("reports/experiments.jsonl").log(result, name="baseline")

summary = jupyter_summary(result)
plot_configuration_chart(result)
```

These helpers keep publication-oriented exports, notebook inspection, and run metadata close to the analysis object.

---

## Architecture

```text
qca
├── __init__.py
├── _constants.py
├── _types.py
├── _version.py
│
├── calibration
│   ├── __init__.py
│   ├── _validators.py
│   ├── crisp.py
│   ├── logistic.py
│   └── piecewise.py
│
├── core
│   ├── __init__.py
│   ├── conditions.py
│   ├── literals.py
│   └── results.py
│
├── engines
│   ├── __init__.py
│   ├── _helpers.py
│   ├── base.py
│   ├── csqca.py
│   ├── fsqca.py
│   ├── mvqca.py
│   └── mixed.py
│
├── minimizers
│   ├── __init__.py
│   ├── algorithms.py
│   ├── backends.py
│   ├── benchmark.py
│   ├── engine.py
│   ├── implicant.py
│   ├── qmc.py
│   ├── remainder.py
│   ├── set_cover.py
│   └── standard.py
│
├── reporting
│   ├── __init__.py
│   ├── _formatters.py
│   ├── export.py
│   ├── jupyter.py
│   ├── report.py
│   └── reproducibility.py
│
├── results
│   ├── __init__.py
│   └── solution.py
│
├── sweep
│   ├── __init__.py
│   ├── anchors.py
│   └── threshold.py
│
└── viz
    ├── __init__.py
    ├── _helpers.py
    ├── bar_plots.py
    ├── configuration_chart.py
    ├── coverage.py
    ├── dashboard.py
    ├── distribution.py
    ├── truth_table.py
    └── xy_plot.py
```

---

Current engine architecture:

* `QCAEngineBase` holds the shared validation, truth-table, minimization, and
  result-building core.
* `CSQCA`, `FSQCA`, and `MVQCA` are first-class engines in their own modules:
  `qca.engines.csqca`, `qca.engines.fsqca`, and `qca.engines.mvqca`.
* `MixedQCA` is a high-level facade over the same shared core for workflows
  combining crisp, fuzzy, and multi-value conditions.
* `qca.core` owns the shared condition schema, literal objects, and result
  dataclasses; engine classes are exposed only from `qca.engines` and the
  package-level API.
* Private modules prefixed with `_` contain shared implementation details and
  are not part of the stable public API.

## Design Principles

### 1. Python-native

PyQCA is designed for Python workflows from the ground up.

```python
import pandas as pd
from qca import FSQCA
```

The library should work naturally with pandas DataFrames, Jupyter notebooks, Python visualization tools, and scientific computing libraries.

---

### 2. Methodologically transparent

PyQCA should expose intermediate objects such as:

* calibrated data;
* truth tables;
* candidate configurations;
* minimized solutions;
* consistency and coverage scores;
* case-level coverage;
* threshold-sweep results.

QCA should not behave like a black box.

---

### 3. Mixed-type by design

Many real datasets contain a mixture of crisp, categorical, ordinal, and fuzzy-set variables.

PyQCA treats this as a first-class use case through `MixedQCA`.

---

### 4. Pluggable algorithms

Minimization is treated as a replaceable backend.

This enables comparisons between standard Boolean minimization, Quine-McCluskey-style minimization, set-covering-based minimization, and future optimization-based approaches.

---

### 5. Thresholds as analytical variables

Threshold choices can strongly influence QCA results.

PyQCA’s sweep layer is designed to help researchers examine how results change across reasonable threshold spaces instead of relying on a single fixed calibration choice.

---

## Installation

PyQCA is not yet published on PyPI.

For development installation:

```bash
git clone https://github.com/t-yamsaki/PyQCA.git
cd pyqca
pip install -e .
```

Planned PyPI installation:

```bash
pip install pyqca
```

---

## Basic Usage

```python
import pandas as pd
from qca import MixedQCA

df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "A": [1, 1, 1, 1, 0, 0, 0, 0],
        "B": ["high", "high", "low", "low", "high", "high", "low", "low"],
        "C": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        "Y": [0.95, 0.85, 0.80, 0.70, 0.40, 0.30, 0.20, 0.10],
    }
)

model = MixedQCA(
    data=df,
    case_id="case",
    outcome="Y",
    conditions=["A", "B", "C"],
    condition_types={
        "A": "crisp",
        "B": "multi",
        "C": "fuzzy",
    },
)

result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
    minimizer="standard",
)

print(result.summary())
```

---

## Result Objects

PyQCA result objects are designed to be inspectable and exportable.

```python
result.truth_table
result.solutions
result.formula
result.formulas
result.consistency
result.coverage
result.case_coverage
result.condition_schema
result.to_dataframe()
result.to_formula()
result.to_formulas()
result.export_formula("solution.txt")
result.to_markdown()
```

Threshold-sweep results provide additional fields:

```python
sweep_result.summary_df
sweep_result.settings
sweep_result.solutions
sweep_result.truth_tables
sweep_result.stability
sweep_result.to_markdown()
sweep_result.plot_heatmap()
sweep_result.plot_trajectory()
```

---

## Roadmap

### v0.1 — Core QCA

* [x] Data validation
* [x] Crisp-set calibration
* [x] Fuzzy-set calibration
* [x] Multi-value condition handling
* [x] Truth table construction
* [x] Necessity analysis
* [x] Sufficiency analysis
* [x] Consistency and coverage metrics
* [x] Basic csQCA
* [x] Basic mvQCA
* [x] Basic fsQCA

### v0.2 — MixedQCA

* [x] Unified mixed-type condition schema
* [x] `MixedQCA` high-level API
* [x] Mixed crisp / multi-value / fuzzy workflows
* [x] Unified result object
* [x] Markdown export
* [x] Formula export

### v0.3 — Minimization Backends

* [x] Standard Boolean minimizer
* [x] Quine-McCluskey-style minimizer
* [x] Set-covering-based minimizer
* [x] Greedy set-cover minimizer
* [x] Exact set-cover backend
* [x] Benchmarking utilities

### v0.4 — ThresholdSweep

* [x] Outcome threshold sweep
* [x] Single-condition threshold sweep
* [x] Multi-condition threshold grid search
* [x] Dual outcome-condition threshold sweep
* [x] Sweep result summaries
* [x] Heatmap and trajectory visualization

### v0.5 — fsQCA Anchor Sensitivity

* [x] Full membership anchor sweep
* [x] Crossover anchor sweep
* [x] Full non-membership anchor sweep
* [x] Multi-anchor grid analysis
* [x] Calibration stability reports

### v0.6 — Reporting and Reproducibility

* [x] Automated Markdown reports
* [x] LaTeX table export
* [x] Configuration charts
* [x] Reproducibility metadata
* [x] Experiment logging
* [x] Jupyter-friendly summaries

---

## Relationship to Existing QCA Tools

PyQCA is inspired by the broader QCA software ecosystem, especially mature R-based workflows.

Its intended position is:

| Tool / Approach         | Main Role                                                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| R `QCA` package         | Established QCA workflow in R                                                                                         |
| scpQCA-style algorithms | Set-covering-based alternative minimization                                                                           |
| ThS-QCA framework       | Threshold-sweep sensitivity analysis                                                                                  |
| PyQCA                   | Python-native QCA platform integrating mvQCA, fsQCA, mixed-type analysis, minimization backends, and threshold sweeps |

PyQCA aims to complement existing tools, not replace them.

---

## Research Use Cases

PyQCA is designed for research workflows such as:

* configurational analysis in social science;
* management and organization studies;
* policy evaluation;
* marketing and advertising strategy analysis;
* small-N and medium-N causal analysis;
* robustness and sensitivity analysis of QCA results;
* comparison of minimization algorithms;
* QCA-based interpretable rule discovery;
* integration with machine learning pipelines.

---

## Example Research Workflow

```python
from pathlib import Path

import pandas as pd

from qca import MixedQCA, ThresholdSweep

# 1. Analyze calibrated mixed-type conditions.
mixed_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "Strategy": ["focus", "focus", "broad", "broad"] * 2,
        "Budget": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        "CreativeQuality": [0.8, 0.9, 0.6, 0.7, 0.3, 0.4, 0.1, 0.2],
        "MarketFit": [1, 1, 1, 0, 1, 0, 0, 0],
        "HighPerformance": [0.95, 0.90, 0.80, 0.65, 0.45, 0.35, 0.20, 0.10],
    }
)

model = MixedQCA(
    data=mixed_df,
    case_id="case",
    outcome="HighPerformance",
    conditions=["Strategy", "Budget", "CreativeQuality", "MarketFit"],
    condition_types={
        "Strategy": "multi",
        "Budget": "fuzzy",
        "CreativeQuality": "fuzzy",
        "MarketFit": "crisp",
    },
)

# 2. Run standard QCA
result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
    minimizer="standard",
)

# 3. Run set-covering minimization
scp_result = model.fit(
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
    minimizer="set_cover",
)

# 4. Run threshold sweep
raw_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 13)],
        "Strategy": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2],
        "Budget": [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 2],
        "CreativeQuality": [9, 7, 8, 6, 7, 5, 6, 4, 5, 3, 4, 2],
        "MarketFit": [8, 9, 7, 8, 6, 7, 5, 6, 4, 5, 3, 2],
        "HighPerformance": [10, 9, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2],
    }
)

sweep = ThresholdSweep(
    raw_df,
    outcome="HighPerformance",
    conditions=["Strategy", "Budget", "CreativeQuality", "MarketFit"],
    case_id="case",
)

sweep_result = sweep.outcome(
    thresholds=[6, 7, 8, 9],
    condition_thresholds={
        "Strategy": 7,
        "Budget": 7,
        "CreativeQuality": 7,
        "MarketFit": 7,
    },
    incl_cut=0.8,
    n_cut=1,
    minimizer="set_cover",
)

# 5. Export report
Path("reports").mkdir(parents=True, exist_ok=True)
sweep_result.to_markdown("reports/threshold_sweep.md")
```

---

## Citation

If you use PyQCA in academic work, please cite the software using
[`CITATION.cff`](CITATION.cff).

Suggested citation:

```bibtex
@software{pyqca,
  title = {PyQCA: A Python-native toolkit for mvQCA, fsQCA, mixed-type QCA, and threshold-sweep analysis},
  author = {taishi-yamasaki},
  year = {2026},
  url = {https://github.com/t-yamsaki/PyQCA}
}
```

PyQCA’s threshold-sweep layer is inspired by the ThS-QCA framework:

```bibtex
@article{toyoda2026thsqca,
  title = {ThS-QCA: Threshold-Sweep Qualitative Comparative Analysis in R},
  author = {Toyoda, Yuki},
  year = {2026},
  journal = {arXiv preprint arXiv:2601.11229}
}
```

---

## License

This project is released under the MIT License.

See [`LICENSE`](LICENSE) for the project license,
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution, and
[`PROVENANCE_AUDIT.md`](PROVENANCE_AUDIT.md) for the source-origin review.

---

## Contributing

Contributions are welcome.

Contributions are accepted under the project's MIT License. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for development, attribution, and
provenance requirements.

Potential contribution areas:

* QCA algorithm implementation;
* minimization backends;
* benchmarking against R QCA outputs;
* fsQCA calibration utilities;
* threshold-sweep visualization;
* documentation and tutorials;
* examples from applied research.

Please open an issue before submitting a large pull request.

---

## Project Status

PyQCA is currently experimental.

The initial development priority is methodological correctness and reproducibility rather than API stability. Breaking changes may occur before v1.0.

Recommended use for now:

* exploratory research;
* prototype analysis;
* benchmarking;
* educational examples;
* methodological experimentation.

Not yet recommended for production or high-stakes applied research without manual validation.
