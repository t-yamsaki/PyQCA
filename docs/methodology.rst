Methodology and attribution
===========================

PyQCA is a software implementation, not a substitute for methodological
judgment. Researchers remain responsible for calibration decisions,
directional expectations, truth-table thresholds, remainder handling,
interpretation, and robustness assessment.

Related methods
---------------

Core QCA
   PyQCA uses established QCA concepts including truth tables, necessity,
   sufficiency, complex/parsimonious/intermediate solutions, consistency, and
   coverage. The R ``QCA`` package and Dusa's published work are important
   methodological references.

Threshold sweep
   :class:`qca.ThresholdSweep` is a Python-native adaptation of the
   algorithmic structure published by ThSQCA. Its MIT attribution is preserved
   in ``THIRD_PARTY_NOTICES.md``.

Machine-learning-enhanced QCA
   :mod:`qca.mlqca` is methodologically informed by Huang's mlQCA protocol.
   PyQCA independently implements the workflow using its own data contracts,
   QCA engines, search, reporting, and stability analysis.

Source and license audit
------------------------

See the repository files:

* ``THIRD_PARTY_NOTICES.md`` for third-party attribution;
* ``PROVENANCE_AUDIT.md`` for the source-origin review;
* ``CITATION.cff`` for software citation metadata;
* ``LICENSE`` for the MIT License.

Key mlQCA article
-----------------

Qin Huang (2025), "Towards machine-learning enhanced QCA: optimizing coverage
and empirical significance", *Quality & Quantity*, 59, 4259-4281.
https://doi.org/10.1007/s11135-025-02146-2
