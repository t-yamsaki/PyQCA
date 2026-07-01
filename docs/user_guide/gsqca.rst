gsQCA
=====

:class:`qca.GSQCA` is the generalized-set QCA engine for datasets containing
crisp, multi-value, and fuzzy conditions. PyQCA operationalizes this as a
common truth-table and minimization workflow where csQCA, mvQCA, and fsQCA can
be treated as special cases. This page documents PyQCA's implemented
semantics rather than claiming one-to-one compatibility with any single
external gsQCA package.

Explicit condition types
------------------------

.. code-block:: python

   from qca import GSQCA

   model = GSQCA(
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

   model = GSQCA.from_schema(
       data=data,
       outcome="Y",
       schema=schema,
       case_id="case",
   )

The normalized schema is available through ``model.condition_schema`` and the
detected workflow through ``model.workflow``.

Multivalent fuzzy set variables
-------------------------------

For generalized-set QCA, a multi-value condition can also be represented by
value-specific calibrated membership columns. Use ``value_columns`` to map
each domain value to its membership column:

.. code-block:: python

   schema = [
       {
           "name": "sector",
           "type": "multi-value",
           "calibrated": True,
           "value_columns": {
               "public": "sector_public",
               "private": "sector_private",
               "hybrid": "sector_hybrid",
           },
       },
       {
           "name": "capacity",
           "type": "fuzzy",
           "calibrated": True,
       },
   ]

   model = GSQCA.from_schema(
       data=data,
       outcome="Y",
       schema=schema,
       case_id="case",
   )

``GSQCA`` uses the value-specific memberships when computing row consistency,
necessity, sufficiency, and minimized solution membership. Truth-table case
assignment uses the strongest declared value set for each multi-value
condition; consistency is computed over all cases from the original calibrated
memberships.
