Calibration
===========

Calibration converts raw measures into set-membership scores. PyQCA provides
crisp, piecewise fuzzy, and logistic fuzzy calibration utilities.

Crisp calibration
-----------------

.. code-block:: python

   from qca import calibrate_crisp

   calibrated = calibrate_crisp(
       data,
       value_col="raw_policy",
       threshold=5,
       out_col="policy",
       direction="ge",
   ).df

The generic crisp API accepts ``ge``, ``gt``, ``le`` and ``lt`` (and their
comparison-symbol aliases). Equality is included by ``ge``/``le`` only.
The ``high``/``low`` aliases belong to the separate mlQCA proposal API.

Piecewise fuzzy calibration
---------------------------

.. code-block:: python

   from qca import calibrate_piecewise

   calibrated = calibrate_piecewise(
       data,
       value_col="raw_capacity",
       out_col="capacity",
       quantiles=(0.1, 0.5, 0.9),
   ).df

This DataFrame API estimates anchors from the specified quantiles. For fixed
raw anchors, use the Series API (it is not an anchors argument on
calibrate_piecewise):

.. code-block:: python

   from qca.calibration import piecewise_fuzzy_series

   calibrated = data.copy()
   calibrated["capacity"] = piecewise_fuzzy_series(
       data["raw_capacity"], p_low=2, p_mid=5, p_high=8,
   )

The three anchors map to lower membership, 0.5, and upper membership;
the default bounds are 0.05 and 0.95.

Logistic fuzzy calibration
--------------------------

.. code-block:: python

   from qca import calibrate_logistic

   calibrated = calibrate_logistic(
       data,
       value_col="raw_support",
       out_col="support",
       full_out=2,
       crossover=5,
       full_in=12,
   ).df

Calibration is a substantive analytical decision. Use
:doc:`sensitivity` to inspect whether conclusions change across reasonable
thresholds or fuzzy anchors.

Three-anchor logistic semantics
-------------------------------

Logistic calibration uses two log-odds slopes joined continuously at the
crossover (membership 0.5). Below the crossover the slope is
``-logit(mem_full_out) / (crossover - full_out)``; above it the slope is
``logit(mem_full_in) / (full_in - crossover)``. Each segment uses
``z = slope * (x - crossover)`` and membership ``1 / (1 + exp(-z))``.
Thus anchors ``(0, 1, 10)`` map to ``(0.05, 0.5, 0.95)`` by default.
The curve is continuous and increasing, but need not have equal derivatives
on both sides. See `Dușa, QCA with R, Calibration
<https://bookdown.org/dusadrian/QCAbook/calibration.html>`_.

Custom targets must satisfy ``0 < mem_full_out < 0.5 < mem_full_in < 1``.
Raw anchors must be finite and strictly increasing, with representable positive
segment slopes. Missing and nonfinite input observations remain missing.
Numerically saturated finite tails stay inside (0, 1) at floating-point limits.
The same calibration applies to explicit and quantile-derived anchors.

``LogisticAnchor.slope_below`` and ``slope_above`` (also in ``to_dict()``)
store the actual centered slopes. Reconstruct the curve using these values
and ``crossover``. The legacy ``slope`` and ``intercept`` describe a single
curve only when the slopes agree to numerical precision; otherwise both are
``None``. Existing positional construction is retained, but callers consuming
single-curve metadata must handle ``None`` for asymmetric calibration.
The exported legacy helper ``_estimate_logistic_params`` rejects curves
requiring two slopes instead of returning misleading single-curve parameters.

Piecewise boundary and grouping rules
-------------------------------------

Piecewise membership bounds must be finite and satisfy ``0 <= lower < 0.5 < upper <=
1``. Bounds exactly equal to 0.5 are rejected to avoid a flat segment at the crossover;
endpoints 0 and 1 are allowed. Scalar, Series, and quantile-based piecewise calibration
apply the same validation and raise ``ValueError`` for invalid bounds. Logistic
calibration retains its separate, stricter endpoint requirements.

Piecewise calibration uses the actual positive anchor intervals, without imposing an
absolute minimum width. Direct scalar/Series calls require finite, strictly increasing
anchors and raise ``ValueError`` for degenerate anchors; quantile calibration retains
its existing warning-and-skip policy for invalid estimated anchors. Unit conversion
preserves memberships within floating-point precision while the anchors remain distinct
and finite. Values that round to identical floating-point anchors cannot be
distinguished; no artificial interval is substituted.

Grouped ``calibrate_piecewise`` calibration uses each group's actual rows, even when
index labels are duplicated within or across groups. Returned rows retain the input
order and index; anchor counts, ``min_nonnull``, and calibrated/skipped counts reflect
rows rather than unique labels.
