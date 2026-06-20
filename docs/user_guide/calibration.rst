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
       crossover=5,
   ).df

Calibration is a substantive analytical decision. Use
:doc:`sensitivity` to inspect whether conclusions change across reasonable
thresholds or fuzzy anchors.
