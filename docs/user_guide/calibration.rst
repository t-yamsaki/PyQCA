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
       direction="high",
   ).df

Piecewise fuzzy calibration
---------------------------

.. code-block:: python

   from qca import calibrate_piecewise

   calibrated = calibrate_piecewise(
       data,
       value_col="raw_capacity",
       out_col="capacity",
       anchors=(2, 5, 8),
   ).df

The anchors represent full non-membership, crossover, and full membership.
When anchors are omitted, the utility can estimate them from quantiles.

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
