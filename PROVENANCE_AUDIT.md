# Source Provenance Audit

Audit date: 2026-06-19

Scope:

- `src/qca/sweep/threshold.py`
- `src/qca/minimizers/`
- `src/qca/mlqca/`
- the former private MixedQCA project used as the starting point for PyQCA

This is a source-provenance review for release preparation. It is not a legal
opinion.

## Conclusion

PyQCA may remain under the MIT License, subject to preserving the ThSQCA MIT
notice in `THIRD_PARTY_NOTICES.md`.

- ThresholdSweep is treated as an MIT-licensed Python adaptation of ThSQCA.
- No direct source translation from the GPL-licensed R package QCA was found in
  the PyQCA minimizer implementation.
- The mlQCA implementation uses the published protocol and public package API
  as methodological references without translating R source code.
- The former private MixedQCA project was authored by the PyQCA copyright
  holder and introduces no third-party licensing requirement.

## Materials Reviewed

### PyQCA

- Repository: https://github.com/t-yamsaki/PyQCA
- Local version: 0.2.0
- License: MIT

### ThSQCA

- Repository: https://github.com/im-research-yt/ThSQCA
- Reviewed commit: `c51aff21f3110a306c0358370ed7cbbb1e44e796`
- Package version: 2.0.0
- License: MIT
- Files reviewed:
  - `R/tsqca_core.R`
  - `R/tsqca_ots_dts.R`
  - `R/tsqca_cts.R`

### R package QCA

- CRAN package version reviewed: 3.25
- Repository: https://github.com/dusadrian/QCA
- License: GPL-3.0-or-later
- Files and implementation areas reviewed:
  - `R/minimize.R`
  - `R/makeChart.R`
  - `R/solveChart.R`
  - `src/QCA.c`
  - `src/CCubes.c`
  - set-cover and prime-implicant chart C sources

## ThresholdSweep Findings

The following correspondences with ThSQCA are intentional:

| ThSQCA | PyQCA |
| --- | --- |
| `qca_bin(x, thr)` | `qca_bin(x, threshold)` |
| `validate_pre_calibrated()` | `validate_pre_calibrated()` |
| `prepare_dat_bin()` | `prepare_threshold_data()` |
| `otSweep()` | `ThresholdSweep.outcome()` |
| `ctSweepS()` | `ThresholdSweep.condition()` |
| `ctSweepM()` | `ThresholdSweep.multi_condition()` |
| `dtSweep()` | `ThresholdSweep.dual()` |
| `thrY`, `expression`, `inclS`, `covS` | matching summary concepts |

Both implementations:

1. binarize values using `x >= threshold`;
2. always threshold the outcome;
3. optionally pass calibrated conditions through unchanged;
4. run one QCA analysis per threshold combination;
5. collect expressions and fit metrics.

PyQCA differs materially in implementation:

- Python/pandas dataclasses replace R data frames and S3 result classes;
- PyQCA invokes its own `MixedQCA` and minimization backends rather than
  `QCA::truthTable()` and `QCA::minimize()`;
- validation, exception handling, result storage, plotting, and Markdown
  generation are independently implemented;
- PyQCA adds unified result objects and fsQCA anchor-sensitivity integration.

Because names, workflow structure, and portions of validation behavior closely
track ThSQCA, PyQCA conservatively treats this module as an adaptation under
ThSQCA's MIT license and preserves its attribution.

## Minimizer Findings

PyQCA's minimizers use:

- Python tuple-based implicant patterns;
- iterative Quine-McCluskey candidate merging;
- explicit remainder classification;
- a deterministic greedy set-cover selector;
- a Python branch-and-bound exact set-cover selector.

R QCA 3.25 uses a substantially different implementation:

- R orchestration around matrix representations;
- compiled C routines such as `C_QMC`;
- prime-implicant chart and matrix reduction routines;
- bundled or integrated solver-oriented C code.

An identifier comparison across PyQCA sweep/minimizer functions found:

- intentional ThSQCA overlap: `qca_bin`, `validate_pre_calibrated`;
- R QCA overlap: `complexity`, `minimize`, both generic algorithm terms.

No copied R/C identifiers, comments, error text, data structures, or
line-by-line control flow were identified in the PyQCA minimizer modules.
Quine-McCluskey and set cover are published algorithms and were implemented in
PyQCA using Python-native data structures.

## MixedQCA Origin

The former MixedQCA project was private and authored by the same copyright
holder as PyQCA. It is first-party code and therefore does not require a
third-party notice.

## mlQCA Findings

PyQCA's `qca.mlqca` package is a Python-native implementation. The P0-P6
modules define PyQCA-specific configuration, validation, result contracts, an
optional XGBoost backend, feature-importance summaries, split-threshold
aggregation, crisp calibration, csQCA evaluation, condition-combination
search, Pareto analysis, fuzzy-anchor proposals, fsQCA calibration, and anchor
sensitivity adapters. MixedQCA integration, Markdown reporting,
visualization, and reproducibility metadata are also independently
implemented with PyQCA-native components. P6 adds stratified cross-validation,
stratified bootstrap with out-of-bag evaluation, and aggregate feature and
cutoff stability tables.

The implementation was written against:

- the protocol described by Huang (2025);
- the public mlQCA website and function-level documentation;
- public XGBoost APIs including `pred_contribs`, `get_score()`, and
  `trees_to_dataframe()`.

No R source from the mlQCA package was copied or translated. PyQCA's data
models, error handling, ranking tables, optional dependency behavior, and test
suite are independently implemented.

The reproduction test fixture `tests/data/mlqca_vote_data.csv` is a mechanical
conversion of `data/voteData.rda` from mlQCA repository commit
`f5bce883ace6bdae394853539aca66d6576deba1`. The source repository is MIT
licensed. No analytical transformation was applied during conversion.

## Release Requirements

Before each public release:

1. retain `LICENSE` and `THIRD_PARTY_NOTICES.md` in source and wheel artifacts;
2. update this audit when code is copied or substantially adapted from another
   project;
3. verify licenses before adding bundled code or runtime dependencies;
4. require contributors to submit code under the project's MIT License;
5. avoid copying GPL source into the MIT-licensed codebase.
