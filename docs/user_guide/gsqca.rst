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

Column names and membership resolution
--------------------------------------

Truth-table grouping uses temporary column names that do not collide with input columns.
Condition names beginning with ``__crisp_`` or ``__gsqca_corner_`` are supported without
changing case assignments or minimization results.

Explicit ``value_columns`` aliases are resolved from the original input before any
canonical columns are written. Swapped aliases, shared source columns, and sources used
by other explicitly mapped conditions are allowed and independent of mapping/condition
order. A target that collides with the outcome, case ID, a condition name, or another
value-membership target (including an implicitly resolved value) raises ``ValueError``.
Explicit aliases still take precedence over competing conventional columns for the same
value. All source columns must exist in the original input; the caller's DataFrame is
not modified.

Multi-value engines resolve memberships using the same condition schema as truth-table
assignment. A categorical condition uses its category column even when auxiliary columns
named ``A=0``, ``A[0]``, or ``A__0`` exist. GSQCA uses value memberships when
``value_columns`` is declared, ``calibrated=True`` is specified with value columns, or
only value columns are present. The standalone ``MultiValueLiteral.membership(data)``
API has no engine schema and retains its value-column-first behavior.

Solution formulas and coverage-table labels preserve multi-value categories explicitly,
including ``A=0``, ``A=1``, and ``A=2``. Crisp/fuzzy set conditions retain ``A`` and
``~A`` notation. Low-level ``Implicant.label`` and ``build_coverage_table`` accept
optional ``is_multivalue`` flags; callers without schema metadata retain legacy integer
0/1 set notation.
