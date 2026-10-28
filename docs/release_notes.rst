Release notes
=============

0.3.0 (release preparation)
---------------------------

This release collects correctness fixes since the v0.2.0 tag, through PR #60.
It remains experimental. Re-run saved analyses when upgrading: corrected
metrics, calibration and minimization can change truth tables and solutions.
This page describes the prepared release; it does not imply a PyPI upload.

Core QCA and minimization
~~~~~~~~~~~~~~~~~~~~~~~~~

* PRI uses the triple intersection and per-case differences, preserving small
  positive denominators. Necessity and sufficiency retain positive membership
  sums even below earlier absolute tolerances.
* Exact 0.5 condition memberships are excluded from truth-table assignment
  and counts, while all cases still contribute to fuzzy metrics.
* Truth-table inclusion applies frequency, consistency and PRI cutoffs.
  Low-frequency configurations remain eligible logical remainders.
* Necessary-condition triviality uses explicit universal-membership evidence;
  manually constructed results without it can report an unknown value.
* The standard backend now selects an exact minimum cover (terms first, then
  logical complexity). Approximation requires greedy_set_cover explicitly.
* Intermediate solutions use easy counterfactuals bounded by the selected
  parsimonious region, preserving the complex/intermediate/parsimonious subset
  relation. Direction names and domain values are validated even for empty
  positive sets.
* Multi-value membership resolution respects engine schemas; formulas retain
  category labels. Wildcard category values and unsafe alias collisions are
  rejected. Temporary grouping columns no longer collide with user names.

See :doc:`user_guide/core_qca`, :doc:`user_guide/gsqca`, and
:doc:`user_guide/minimization` for precise definitions and limitations.

Calibration and sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Logistic calibration honors asymmetric anchors using separate centered
  slopes. Legacy single-slope metadata can be None for asymmetric curves.
* Piecewise calibration uses actual positive anchor spacing, requires bounds
  to straddle 0.5, and groups duplicate-index rows by position.
* Threshold preparation preserves missing/nonfinite values; affected sweep
  runs record errors rather than silently treating them as absent membership.
* Threshold and anchor sweeps reject reserved outcome/output name collisions.
* Anchor direction mappings accept original or calibrated names and reject
  conflicting aliases.

See :doc:`user_guide/calibration` and :doc:`user_guide/sensitivity`.

Machine-learning-enhanced QCA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* XGBoost feature keys are resolved using model metadata, avoiding confusion
  between literal feature names and positional aliases.
* SHAP direction inference is stable under positive unit rescaling, subject
  to floating-point representation and equivalent model contributions.
  Unidentified directions require explicit specification; mean SHAP sign is
  no longer a fallback.
* Crisp and fuzzy calibration use normalized direction aliases for both
  calculations and returned metadata, preserving the caller's input table.
* Anchor sweeps accept original feature names, including low-direction
  features, with expectations expressed on the same calibrated set.

See :doc:`user_guide/mlqca` for migration guidance.

Distribution and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package metadata, runtime version and citation identify 0.3.0.
English and Japanese READMEs and Sphinx guides describe the corrected
semantics. Python 3.11 and 3.12 remain the CI targets. The base dependencies
remain NumPy and pandas; ML and visualization dependencies are optional.

0.2.0
-----

The previous repository tag is v0.2.0. It provides the baseline core and
generalized-set engines, minimizers, calibration, sensitivity, reporting
and optional mlQCA workflow. A repository tag alone does not establish
availability on PyPI.
