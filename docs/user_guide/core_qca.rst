Core QCA engines
================

PyQCA provides first-class engines for the three core QCA families. Each
engine validates the condition representation expected by that method while
sharing truth-table, necessity, sufficiency, minimization, and result-building
infrastructure.

csQCA
-----

:class:`qca.CSQCA` requires binary crisp-set conditions.

.. code-block:: python

   from qca import CSQCA

   model = CSQCA(
       data=data,
       outcome="Y",
       conditions=["A", "B", "C"],
       case_id="case",
   )
   result = model.fit(consistency_cutoff=0.8)

mvQCA
-----

:class:`qca.MVQCA` accepts categorical or ordinal multi-value conditions.

.. code-block:: python

   from qca import MVQCA

   model = MVQCA(
       data=data,
       outcome="Y",
       conditions=["institution", "strategy"],
       case_id="case",
   )
   result = model.fit(consistency_cutoff=0.8)

fsQCA
-----

:class:`qca.FSQCA` accepts calibrated memberships in the closed interval
``[0, 1]``. Crisp conditions can be included in an fsQCA model by declaring
their condition type.

.. code-block:: python

   from qca import FSQCA

   model = FSQCA(
       data=data,
       outcome="Y",
       conditions=["capacity", "support"],
       condition_types={"capacity": "fuzzy", "support": "crisp"},
       case_id="case",
   )
   result = model.fit(
       consistency_cutoff=0.8,
       outcome_threshold=0.5,
   )

Common analysis methods
-----------------------

All core engines expose methods for:

* truth-table construction with :meth:`qca.QCAEngineBase.build_truth_table`;
* necessity analysis with :meth:`qca.QCAEngineBase.evaluate_necessity`;
* sufficiency analysis with :meth:`qca.QCAEngineBase.evaluate_sufficiency`;
* configuration search with
  :meth:`qca.QCAEngineBase.search_sufficient_configurations`;
* unified fitting and minimization with :meth:`qca.QCAEngineBase.fit`.
