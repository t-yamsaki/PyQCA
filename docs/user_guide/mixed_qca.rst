MixedQCA
========

:class:`qca.MixedQCA` is the high-level facade for datasets containing crisp,
multi-value, and fuzzy conditions. It uses the same shared QCA core as the
first-class csQCA, mvQCA, and fsQCA engines.

Explicit condition types
------------------------

.. code-block:: python

   from qca import MixedQCA

   model = MixedQCA(
       data=data,
       outcome="Y",
       conditions=["policy", "region", "capacity"],
       condition_types={
           "policy": "crisp",
           "region": "multi",
           "capacity": "fuzzy",
       },
       case_id="case",
   )

   result = model.fit(
       consistency_cutoff=0.8,
       minimizer="set_cover",
   )

Unified schema
--------------

For reusable pipelines, define the condition schema explicitly:

.. code-block:: python

   schema = [
       {
           "name": "policy",
           "type": "crisp",
           "domain": [0, 1],
           "calibrated": True,
       },
       {
           "name": "region",
           "type": "multi-value",
           "domain": ["north", "south"],
       },
       {
           "name": "capacity",
           "type": "fuzzy",
           "domain": [0, 1],
           "calibrated": True,
       },
   ]

   model = MixedQCA.from_schema(
       data=data,
       outcome="Y",
       schema=schema,
       case_id="case",
   )

The normalized schema is available through ``model.condition_schema`` and the
detected workflow through ``model.workflow``.
