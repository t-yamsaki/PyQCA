# PyQCA

**PyQCA** is a Python-native toolkit for **Qualitative Comparative Analysis (QCA)**.

It provides a unified framework for running **multi-value QCA (mvQCA)**, **fuzzy-set QCA (fsQCA)**, and mixed-type configurational analysis in Python. PyQCA also supports pluggable minimization backends, including standard Boolean minimization and set-covering-based minimization, as well as threshold-sweep analysis inspired by the ThS-QCA framework.

> Status: experimental / under active development

---

## Why PyQCA?

QCA is widely used in social science, management research, political science, evaluation research, and configurational causal analysis. Mature QCA tooling exists in R, especially for crisp-set, multi-value, and fuzzy-set QCA workflows.

PyQCA aims to provide a Python-native QCA foundation for researchers and practitioners who want to:

* run mvQCA and fsQCA directly in Python;
* analyze crisp, multi-valued, and fuzzy-set conditions in a unified interface;
* switch between different minimization algorithms;
* inspect consistency, coverage, truth tables, and solution formulas;
* conduct threshold-sensitivity analysis systematically;
* integrate QCA workflows with Python data science tools such as pandas, NumPy, scikit-learn, Jupyter, and visualization libraries.

PyQCA is not intended to be only a thin wrapper around existing R packages. The goal is to provide a Python-native architecture for standard QCA workflows and future methodological extensions.

---

## Core Features

### 1. mvQCA

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

### 2. fsQCA

Run fuzzy-set QCA with calibrated set-membership scores.

```python
from qca import FSQCA

model = FSQCA(
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

### 3. MixedQCA

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

The purpose of `MixedQCA` is to make mixed-type configurational analysis explicit, rather than forcing all variables into a single representation too early.

---

### 4. Pluggable Minimization Backends

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

Available or planned minimizers:

| Minimizer          | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `standard`         | Standard Boolean minimization backend                |
| `qmc`              | Quine-McCluskey-style minimization                   |
| `set_cover`        | Set-covering-based minimization                      |
| `greedy_set_cover` | Greedy approximation for large candidate rule spaces |
| `exact_set_cover`  | Exact set-covering optimization backend              |

Set-covering-based minimization is designed as an alternative backend, especially useful for large mvQCA-style candidate rule spaces.

---

### 5. ThresholdSweep

PyQCA includes a threshold-sweep analysis layer inspired by the ThS-QCA framework.

Instead of treating thresholds as fixed preprocessing choices, `ThresholdSweep` treats them as explicit analytical variables.

```python
from qca import MixedQCA
from qca.sweep import ThresholdSweep

base = MixedQCA(
    data=df,
    outcome="Y",
    conditions=["X1", "X2", "X3"],
    condition_types={
        "X1": "crisp",
        "X2": "multi",
        "X3": "fuzzy",
    },
)

sweep = ThresholdSweep(base)

result = sweep.outcome(
    thresholds=[6, 7, 8, 9],
    consistency_cutoff=0.8,
    coverage_cutoff=0.1,
)

print(result.summary_df)
```

Planned sweep modes:

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
from qca.sweep import ThresholdSweep

sweep = ThresholdSweep(model)

res = sweep.outcome(
    thresholds=[5, 6, 7, 8, 9],
    minimizer="set_cover",
    consistency_cutoff=0.85,
    coverage_cutoff=0.1,
)

res.summary_df
```

Example output:

| outcome_threshold | n_solutions | best_solution | consistency | coverage |
| ----------------: | ----------: | ------------- | ----------: | -------: |
|                 5 |           3 | A*B + C       |        0.88 |     0.62 |
|                 6 |           2 | A*B           |        0.90 |     0.55 |
|                 7 |           1 | A*B*~C        |        0.92 |     0.41 |
|                 8 |           1 | A*B*D         |        0.91 |     0.33 |
|                 9 |           0 | —             |           — |        — |

This helps researchers inspect how sufficient configurations emerge, disappear, branch, or become more complex as thresholds change.

---

## Example: fsQCA Calibration Anchor Sweep

For fuzzy-set QCA, threshold sensitivity often appears through calibration anchors: full non-membership, crossover, and full membership.

PyQCA plans to support anchor sweep analysis directly.

```python
res = sweep.fuzzy_anchors(
    variable="X3",
    full_out=[1, 2, 3],
    crossover=[5, 6],
    full_in=[8, 9, 10],
    consistency_cutoff=0.8,
)

res.plot_heatmap()
```

This feature is intended to extend threshold-sweep analysis beyond crisp-set dichotomization and into fuzzy-set calibration sensitivity.

---

## Architecture

```text
qca
├── calibration
│   ├── crisp.py
│   ├── fuzzy.py
│   └── multivalue.py
│
├── engines
│   ├── csqca.py
│   ├── mvqca.py
│   ├── fsqca.py
│   └── mixed.py
│
├── minimizers
│   ├── standard.py
│   ├── qmc.py
│   ├── set_cover.py
│   └── greedy.py
│
├── sweep
│   ├── outcome.py
│   ├── condition.py
│   ├── grid.py
│   ├── dual.py
│   └── fuzzy_anchor.py
│
├── results
│   ├── solution.py
│   ├── truth_table.py
│   ├── sweep_result.py
│   └── report.py
│
└── viz
    ├── heatmap.py
    ├── trajectory.py
    └── configuration_chart.py
```

---

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
git clone https://github.com/YOUR_USERNAME/pyqca.git
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

df = pd.read_csv("data.csv")

model = MixedQCA(
    data=df,
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
result.consistency
result.coverage
result.case_coverage
result.to_dataframe()
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

* Data validation
* Crisp-set calibration
* Fuzzy-set calibration
* Multi-value condition handling
* Truth table construction
* Necessity analysis
* Sufficiency analysis
* Consistency and coverage metrics
* Basic mvQCA
* Basic fsQCA

### v0.2 — MixedQCA

* Unified mixed-type condition schema
* `MixedQCA` high-level API
* Mixed crisp / multi-value / fuzzy workflows
* Unified result object
* Markdown export
* Formula export

### v0.3 — Minimization Backends

* Standard Boolean minimizer
* Quine-McCluskey-style minimizer
* Set-covering-based minimizer
* Greedy set-cover minimizer
* Exact set-cover backend
* Benchmarking utilities

### v0.4 — ThresholdSweep

* Outcome threshold sweep
* Single-condition threshold sweep
* Multi-condition threshold grid search
* Dual outcome-condition threshold sweep
* Sweep result summaries
* Heatmap and trajectory visualization

### v0.5 — fsQCA Anchor Sensitivity

* Full membership anchor sweep
* Crossover anchor sweep
* Full non-membership anchor sweep
* Multi-anchor grid analysis
* Calibration stability reports

### v0.6 — Reporting and Reproducibility

* Automated Markdown reports
* LaTeX table export
* Configuration charts
* Reproducibility metadata
* Experiment logging
* Jupyter-friendly summaries

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
from qca import MixedQCA
from qca.sweep import ThresholdSweep

# 1. Define model
model = MixedQCA(
    data=df,
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
sweep = ThresholdSweep(model)

sweep_result = sweep.outcome(
    thresholds=[6, 7, 8, 9],
    minimizer="set_cover",
)

# 5. Export report
sweep_result.to_markdown("reports/threshold_sweep.md")
```

---

## Citation

If you use PyQCA in academic work, please cite this repository.

Suggested citation:

```bibtex
@software{pyqca,
  title = {PyQCA: A Python-native toolkit for mvQCA, fsQCA, mixed-type QCA, and threshold-sweep analysis},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/pyqca}
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

See `LICENSE` for details.

---

## Contributing

Contributions are welcome.

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
