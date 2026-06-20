# Third-Party Notices

PyQCA is distributed under the MIT License. This file records third-party
software and research sources that materially informed the project.

## ThSQCA

PyQCA's threshold-sweep implementation in `qca.sweep.threshold` is a
Python-native adaptation of the algorithmic structure published by ThSQCA.
The adapted concepts include:

- binary calibration using `x >= threshold`;
- pass-through handling for pre-calibrated conditions;
- outcome, single-condition, multi-condition, and dual threshold sweeps;
- summary records containing expressions, consistency, coverage, and solution
  counts.

PyQCA does not call or bundle the R package. It uses its own QCA engines,
result objects, validation, reporting, and visualization implementation.

Project:

- ThSQCA: Threshold-Sweep QCA
- Copyright (c) 2025 Yuki Toyoda
- Repository: https://github.com/im-research-yt/ThSQCA
- CRAN: https://cran.r-project.org/package=ThSQCA
- License: MIT

The ThSQCA MIT license notice follows:

> MIT License
>
> Copyright (c) 2025 Yuki Toyoda
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## R package QCA

PyQCA is methodologically informed by the established R package QCA and uses
standard QCA terminology such as truth tables, complex/parsimonious/intermediate
solutions, directional expectations, consistency, and coverage.

- QCA: Qualitative Comparative Analysis
- Authors include Adrian Dusa, Ciprian Paduraru, and Alrik Thiem
- Repository: https://github.com/dusadrian/QCA
- CRAN: https://cran.r-project.org/package=QCA
- License: GPL-3.0-or-later

No source code from the R package QCA is included in PyQCA. PyQCA does not link
to, import, or distribute that package. The minimization implementation was
reviewed against QCA 3.25; the audit found no direct source translation. See
`PROVENANCE_AUDIT.md`.

## mlQCA

PyQCA's optional `qca.mlqca` workflow is methodologically informed by the
mlQCA project and its accompanying research article. The current Python
implementation independently uses public XGBoost APIs to train a classifier,
rank conditions using prediction contributions and tree importance, and
extract frequently used split thresholds as calibration candidates. PyQCA
then uses its own calibration, csQCA, minimization, model-ranking, and Pareto
evaluation implementations. Fuzzy-set anchor proposals and sensitivity
analysis are implemented using PyQCA's own fsQCA calibration and
`AnchorSensitivity` components. Mixed-type execution, reports,
visualizations, and reproducibility records use PyQCA's own APIs.

- Project: https://qinhuang-poliecon.github.io/mlQCA/
- Repository: https://github.com/qinhuang-poliecon/mlQCA
- License: MIT
- Article: Huang, Q. (2025). *Towards machine-learning enhanced QCA:
  optimizing coverage and empirical significance*. Quality & Quantity, 59,
  4259-4281. https://doi.org/10.1007/s11135-025-02146-2

No R source code from mlQCA is copied, bundled, or executed by PyQCA.

The source distribution includes `tests/data/mlqca_vote_data.csv` solely as a
reproduction-test fixture. It is a mechanical CSV conversion of
`data/voteData.rda` from mlQCA repository commit
`f5bce883ace6bdae394853539aca66d6576deba1`, distributed under that
repository's MIT License. The fixture is not included in the PyQCA wheel.

## Research References

The following works describe methods implemented or discussed by PyQCA. Their
inclusion is scholarly attribution and does not imply that their text or
software is distributed with PyQCA.

- Toyoda, Y. (2026). *TSQCA: Threshold-Sweep Qualitative Comparative Analysis
  in R*. arXiv:2601.11229. https://arxiv.org/abs/2601.11229
- Dusa, A. (2019). *QCA with R: A Comprehensive Resource*. Springer.
  https://doi.org/10.1007/978-3-319-75668-4
- Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond*.
  University of Chicago Press.
  https://doi.org/10.7208/chicago/9780226702797.001.0001
- Quine, W. V. (1952). The Problem of Simplifying Truth Functions.
  *The American Mathematical Monthly*, 59(8), 521-531.
  https://doi.org/10.2307/2308219
- McCluskey, E. J. (1956). Minimization of Boolean Functions.
  *Bell System Technical Journal*, 35(6), 1417-1444.
  https://doi.org/10.1002/j.1538-7305.1956.tb03835.x

## Python Dependencies

PyQCA depends on NumPy and pandas and optionally uses Matplotlib. These
libraries are installed separately and retain their own licenses. Their source
code is not copied into this repository or distributed inside the PyQCA source
tree.

The optional `mlqca` dependency group uses XGBoost (Apache-2.0) and
scikit-learn (BSD-3-Clause). The optional `mlqca-explain` group additionally
uses SHAP (MIT). These packages are installed separately and are not bundled
in PyQCA wheels.
