# PyQCA

**PyQCA** is a Python-native toolkit for **Qualitative Comparative Analysis (QCA)**.

[![CI](https://github.com/t-yamsaki/PyQCA/actions/workflows/ci.yml/badge.svg)](https://github.com/t-yamsaki/PyQCA/actions/workflows/ci.yml)
[![Docs build](https://github.com/t-yamsaki/PyQCA/actions/workflows/docs.yml/badge.svg)](https://github.com/t-yamsaki/PyQCA/actions/workflows/docs.yml)
[![Documentation](https://readthedocs.org/projects/pyqca/badge/?version=latest)](https://pyqca.readthedocs.io/en/latest/)

[日本語 README](README.ja.md)

It provides a unified framework for running **crisp-set QCA (csQCA)**,
**multi-value QCA (mvQCA)**, **fuzzy-set QCA (fsQCA)**,
**generalized-set QCA (gsQCA)**, and **machine-learning-enhanced QCA
(mlQCA)** in Python. PyQCA also supports pluggable minimization backends,
threshold and calibration sensitivity analysis, reporting, and reproducibility
tools.

> Status: experimental / under active development

---

## Documentation

The Sphinx documentation covers installation, core QCA, generalized-set QCA,
calibration, minimization, sensitivity analysis, reporting, mlQCA, and the
generated API reference. The hosted documentation is available at
https://pyqca.readthedocs.io/en/latest/.

Build it locally with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

For local previews, open `docs/_build/html/index.html` after a local build.

---

## Why PyQCA?

QCA is widely used in social science, management research, political science, evaluation research, and configurational causal analysis. Mature QCA tooling exists in R, especially for crisp-set, multi-value, and fuzzy-set QCA workflows.

PyQCA aims to provide a Python-native QCA foundation for researchers and practitioners who want to:

* run csQCA, mvQCA, fsQCA, and gsQCA directly in Python;
* analyze crisp, multi-valued, and fuzzy-set conditions in a generalized-set interface;
* switch between different minimization algorithms;
* inspect consistency, coverage, truth tables, and solution formulas;
* conduct threshold-sensitivity analysis systematically;
* use XGBoost evidence to propose conditions, calibration thresholds, and QCA
  model combinations;
* evaluate mlQCA condition and cutoff stability with cross-validation and
  bootstrap resampling;
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

### 4. GSQCA

`GSQCA` is the high-level interface for generalized-set QCA workflows
containing crisp, multi-value, and fuzzy-set conditions.

It treats csQCA, mvQCA, and fsQCA as special cases. Use `GSQCA` for workflows
that combine condition kinds.
The implemented semantics are documented by PyQCA and are not presented as a
drop-in clone of any single external gsQCA package.

```python
from qca import GSQCA

model = GSQCA(
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

The same workflow can be built from the unified condition schema:

```python
schema = [
    {"name": "A", "type": "crisp", "domain": [0, 1], "calibrated": True},
    {"name": "B", "type": "multi-value", "domain": ["low", "high"]},
    {"name": "C", "type": "fuzzy", "domain": [0, 1], "calibrated": True},
]

model = GSQCA.from_schema(data=df, outcome="Y", schema=schema)

print(model.workflow)          # GSQCA-crisp-fuzzy-multi
print(model.condition_schema)  # normalized schema DataFrame
```

For multivalent fuzzy set variables, declare value-specific calibrated
membership columns:

```python
schema = [
    {
        "name": "sector",
        "type": "multi-value",
        "calibrated": True,
        "value_columns": {
            "public": "sector_public",
            "private": "sector_private",
        },
    },
]
```

Fitted results report `qca_type="GSQCA"`.

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

### 7. Machine-Learning-Enhanced QCA

The optional `qca.mlqca` workflow connects predictive modeling to PyQCA's
native QCA engines. It uses XGBoost to rank candidate conditions and extract
split thresholds, then applies PyQCA calibration, csQCA or gsQCA analysis,
radical or theory-constrained model search, Pareto evaluation, and
cross-validation or bootstrap stability analysis.

```python
from qca import (
    MLQCAConfig,
    fit_xgboost_predictor,
    search_csqca_models,
    validate_mlqca_input,
)

# raw_df contains numeric candidate conditions and a binary outcome Y.
validated = validate_mlqca_input(raw_df, outcome="Y")
config = MLQCAConfig(mode="radical", top_k=10, model_size=4)
predictor = fit_xgboost_predictor(validated, config)
result = search_csqca_models(validated, predictor, config)
```

XGBoost and scikit-learn are optional dependencies. See
[Machine-Learning-Enhanced QCA Workflow](#machine-learning-enhanced-qca-workflow) for the
complete workflow, gsQCA integration, stability analysis, and published
example reproduction.

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

## Machine-Learning-Enhanced QCA Workflow

The optional `qca.mlqca` package implements a Python-native workflow inspired
by the published mlQCA protocol. XGBoost is used to rank candidate conditions
and extract split thresholds; calibration, model search, QCA evaluation,
reporting, and stability analysis are performed by PyQCA.

Install the optional dependencies:

```bash
pip install -e ".[mlqca]"
```

The `mlqca-explain` extra additionally installs SHAP for external exploratory
work. PyQCA's built-in importance table uses XGBoost prediction contributions
and does not require the standalone SHAP package.

### XGBoost, Calibration, and QCA Search

```python
import numpy as np
import pandas as pd

from qca import (
    MLQCAConfig,
    fit_xgboost_predictor,
    search_csqca_models,
    validate_mlqca_input,
)

rng = np.random.default_rng(201)
raw_df = pd.DataFrame(
    {
        "policy": rng.normal(size=80),
        "capacity": rng.normal(size=80),
        "support": rng.normal(size=80),
    }
)
raw_df["Y"] = (
    raw_df["policy"] + 0.5 * raw_df["capacity"] > 0
).astype(int)

validated = validate_mlqca_input(raw_df, outcome="Y")
config = MLQCAConfig(
    mode="radical",
    top_k=3,
    model_size=2,
    random_state=201,
    model_params={
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.2,
    },
)

predictor = fit_xgboost_predictor(validated, config)
result = search_csqca_models(validated, predictor, config)

print(predictor.feature_importance)
print(predictor.cutoff_candidates)
print(result.pareto_models)
print(result.best_qca_result)
```

`mode="radical"` evaluates combinations from the top-ranked conditions.
`mode="conservative"` requires theory-driven constraints through required or
excluded conditions, or explicit `conservative_models`.

Generalized-set workflows combining crisp, fuzzy, and multi-value conditions
can be connected to `GSQCA` with `fit_gsqca_from_predictor()`. Fuzzy anchor
proposals can be evaluated with `run_anchor_sensitivity()`.

### Cross-Validation and Bootstrap Stability

PyQCA provides stratified cross-validation and stratified bootstrap analysis.
Cross-validation evaluates each model on its held-out fold. Bootstrap uses
out-of-bag cases when both outcome classes are available.

```python
from pathlib import Path

from qca import bootstrap_mlqca, cross_validate_mlqca

cv_result = cross_validate_mlqca(
    validated,
    config,
    n_splits=5,
    top_k=3,
)
bootstrap_result = bootstrap_mlqca(
    validated,
    config,
    n_bootstrap=100,
    top_k=3,
)

print(cv_result.run_summary)
print(cv_result.feature_stability)
print(bootstrap_result.cutoff_stability)

Path("reports").mkdir(parents=True, exist_ok=True)
cv_result.to_markdown("reports/mlqca_cv.md")
```

The stability tables report top-condition selection rates, feature-use rates,
rank variation, contribution variation, and rounded cutoff-selection rates.
Use a fixed `random_state` for reproducible resampling.

### Published Example Reproduction

The test suite includes a mechanically converted copy of the public
`voteData` fixture from the mlQCA repository. With seed `201` and a
representative parameter setting from the published search space, the
reproduction test:

* fits all 427 published cases;
* reproduces training accuracy and ROC AUC of `1.0`;
* recovers at least five of the six conditions highlighted by the tutorial
  within PyQCA's top ten conditions;
* confirms the expected 210 four-condition combinations from ten candidates.

This is a protocol-level reproduction test, not a claim that Python XGBoost
must produce byte-identical trees or rankings to the R/caret workflow.
Features near the top-ten boundary can vary across XGBoost and Python runtime
versions. Fixture provenance is documented in `tests/data/README.md` and
`THIRD_PARTY_NOTICES.md`.

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
│   └── gsqca.py
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
├── mlqca
│   ├── __init__.py
│   ├── _optional.py
│   ├── backend.py
│   ├── calibration.py
│   ├── config.py
│   ├── csqca.py
│   ├── cutoffs.py
│   ├── fuzzy.py
│   ├── importance.py
│   ├── gsqca.py
│   ├── reporting.py
│   ├── reproducibility.py
│   ├── results.py
│   ├── schema.py
│   ├── search.py
│   ├── stability.py
│   ├── validation.py
│   ├── viz.py
│   └── xgboost.py
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
* `CSQCA`, `FSQCA`, `MVQCA`, and `GSQCA` are first-class engines in their own
  modules: `qca.engines.csqca`, `qca.engines.fsqca`, `qca.engines.mvqca`, and
  `qca.engines.gsqca`.
* `GSQCA` is the first-class generalized-set engine for workflows that combine
  crisp, fuzzy, and multi-value condition kinds.
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
* threshold-sweep results;
* mlQCA feature-importance and split-threshold tables;
* calibration and anchor proposals;
* candidate-model rankings and Pareto frontiers;
* cross-validation and bootstrap stability tables.

QCA and mlQCA should not behave like black boxes. Predictive evidence,
calibration choices, condition combinations, failed model evaluations, and
resampling stability remain inspectable instead of being hidden behind a
single selected solution.

---

### 3. GSQCA by design

Many real datasets contain crisp, categorical, ordinal, and fuzzy-set variables.

PyQCA treats generalized-set QCA as a first-class use case through `GSQCA`.

---

### 4. Pluggable algorithms

Minimization is treated as a replaceable backend.

This enables comparisons between standard Boolean minimization, Quine-McCluskey-style minimization, set-covering-based minimization, and future optimization-based approaches.

---

### 5. Thresholds as analytical variables

Threshold choices can strongly influence QCA results.

PyQCA’s sweep layer is designed to help researchers examine how results change across reasonable threshold spaces instead of relying on a single fixed calibration choice.

---

### 6. Machine learning as evidence, not authority

PyQCA treats machine learning as an empirical aid for condition screening,
calibration proposals, and model-space exploration. Predictive importance does
not replace theoretical knowledge, causal interpretation, or QCA assumptions.

The `radical` mlQCA mode explores combinations from the ranked condition set.
The `conservative` mode requires theory-driven constraints. Pareto evaluation,
published-example reproduction, and resampling stability are exposed so that
researchers can assess how strongly a proposed configuration depends on one
model fit or one calibration choice.

---

## Installation

PyQCA is not yet published on PyPI.

For development installation:

```bash
git clone https://github.com/t-yamsaki/PyQCA.git
cd pyqca
pip install -e .
```

Install optional feature groups as needed:

```bash
pip install -e ".[viz]"
pip install -e ".[mlqca]"
pip install -e ".[mlqca-explain]"
```

Planned PyPI installation:

```bash
pip install pyqca
```

---

## Basic Usage

```python
import pandas as pd
from qca import GSQCA

df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "A": [1, 1, 1, 1, 0, 0, 0, 0],
        "B": ["high", "high", "low", "low", "high", "high", "low", "low"],
        "C": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        "Y": [0.95, 0.85, 0.80, 0.70, 0.40, 0.30, 0.20, 0.10],
    }
)

model = GSQCA(
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

## Relationship to Existing QCA Tools

PyQCA is inspired by the broader QCA software ecosystem, especially mature R-based workflows.

Its intended position is:

| Tool / Approach         | Main Role                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| R `QCA` package         | Established QCA workflow in R                                                                                                          |
| scpQCA-style algorithms | Set-covering-based alternative minimization                                                                                            |
| ThS-QCA framework       | Threshold-sweep sensitivity analysis                                                                                                   |
| R `mlQCA` package       | Published machine-learning-enhanced QCA protocol using predictive condition selection and calibration evidence                        |
| PyQCA                   | Python-native QCA platform integrating core QCA and GSQCA, minimization, sensitivity analysis, and an independent `qca.mlqca` workflow |

PyQCA aims to complement existing tools, not replace them. Its `qca.mlqca`
package is methodologically informed by Huang's mlQCA work but independently
implemented with PyQCA data models, engines, result objects, reporting, and
stability analysis. See `THIRD_PARTY_NOTICES.md` and `PROVENANCE_AUDIT.md`.

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

from qca import GSQCA, ThresholdSweep

# 1. Analyze calibrated generalized-set conditions.
gsqca_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "Strategy": ["focus", "focus", "broad", "broad"] * 2,
        "Budget": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        "CreativeQuality": [0.8, 0.9, 0.6, 0.7, 0.3, 0.4, 0.1, 0.2],
        "MarketFit": [1, 1, 1, 0, 1, 0, 0, 0],
        "HighPerformance": [0.95, 0.90, 0.80, 0.65, 0.45, 0.35, 0.20, 0.10],
    }
)

model = GSQCA(
    data=gsqca_df,
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
  title = {PyQCA: A Python-native toolkit for core, generalized-set, threshold-sweep, and machine-learning-enhanced QCA},
  author = {Yamasaki, Taishi},
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

When using `qca.mlqca`, please also cite the methodological article:

```bibtex
@article{huang2025mlqca,
  title = {Towards machine-learning enhanced QCA: optimizing coverage and empirical significance},
  author = {Huang, Qin},
  year = {2025},
  journal = {Quality \& Quantity},
  volume = {59},
  pages = {4259--4281},
  doi = {10.1007/s11135-025-02146-2}
}
```

The PyQCA citation identifies the software implementation. The Huang citation
identifies the mlQCA methodology that informed the optional workflow.

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
* mlQCA backends, calibration strategies, and stability diagnostics;
* reproduction studies comparing PyQCA with published mlQCA examples;
* documentation and tutorials;
* examples from applied research.

Contributions to `qca.mlqca` must document methodological sources, dataset
provenance, dependency licenses, and whether code was independently written or
adapted. Do not translate or copy GPL-licensed R source into the MIT-licensed
codebase. Reproduction fixtures must have redistribution-compatible licenses.

Please open an issue before submitting a large pull request.

---

## Project Status

PyQCA is currently experimental.

The current development priority is methodological correctness,
reproducibility, and a clear public API. Breaking changes may occur before
v1.0.

Recommended use for now:

* exploratory research;
* prototype analysis;
* benchmarking;
* educational examples;
* methodological experimentation.

For publications and high-stakes applied research, validate important results
against established QCA software, methodological expectations, and the source
data.

Future work is tracked through
[GitHub Issues](https://github.com/t-yamsaki/PyQCA/issues). Published versions
and release notes will be available through
[GitHub Releases](https://github.com/t-yamsaki/PyQCA/releases).

PRI uses the standard triple intersection of configuration, outcome, and negated outcome. This corrects row selection with `pri_cut`; see [PRI semantics](docs/user_guide/core_qca.rst).

The default `standard` and `qmc` backends now use exact cover selection: fewest terms, then lowest total logical complexity. One optimal cover is returned; equally optimal alternatives are not enumerated. Exact search can take exponential time and never silently falls back to an approximation. Use `greedy_set_cover` explicitly for approximate selection. See [minimization semantics](docs/user_guide/minimization.rst).

Sufficiency metrics now retain positive fuzzy memberships even when every score is at most 0.5. Empirical case-count filters remain separate; see [low-membership sufficiency](docs/user_guide/core_qca.rst).

Logistic calibration now honors asymmetric three-anchor specifications using separate log-odds slopes around the crossover. Custom targets must straddle 0.5; asymmetric curves expose segment coefficients instead of a single slope/intercept. See [calibration semantics](docs/user_guide/calibration.rst).

Cases with a set-condition membership exactly 0.5 are now excluded from truth-table row assignment and frequency counts, with a warning. All original cases still contribute to consistency and PRI. See [crossover semantics](docs/user_guide/core_qca.rst).

Multi-value engines resolve memberships using the same condition schema as truth-table assignment. A categorical condition uses its category column even when auxiliary columns named `A=0`, `A[0]`, or `A__0` exist. GSQCA uses value memberships when `value_columns` is declared, `calibrated=True` is specified with value columns, or only value columns are present. The standalone `MultiValueLiteral.membership(data)` API has no engine schema and retains its value-column-first behavior.
