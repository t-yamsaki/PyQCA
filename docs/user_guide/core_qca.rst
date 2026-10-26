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

PRI consistency
---------------

All engines use the standard proportional reduction in inconsistency (PRI)
formula, with ``T = sum(min(X, Y, 1 - Y))`` and
``PRI = (sum(min(X, Y)) - T) / (sum(X) - T)``. Here ``X`` is the
configuration membership and sums include all cases, not only cases assigned
to its truth-table row. See `Dușa, QCA with R, equation 6.4
<https://www.bookdown.org/dusadrian/QCAbook/analysisofsufficiency.html>`_.

An empty configuration or a numerically zero PRI denominator returns ``0.0``
as a finite convention, rather than evidence of sufficient support. A positive
``pri_cut`` excludes such rows; the default ``pri_cut=0.0`` does not exclude
rows on PRI alone. The ordinary consistency threshold still applies.

Low-membership sufficiency
--------------------------

Sufficiency consistency and coverage use all calibrated memberships, even when
no case has configuration membership above 0.5. For example, ``X=[0.2, 0.4]``
and ``Y=[0.3, 0.5]`` have consistency 1.0 and raw coverage 0.75, with
``n_cases_in=0``. Positive membership sums are not rounded to zero. See
`Ragin (2006), Set Relations in Social Research
<https://sites.socsci.uci.edu/~cragin/fsQCA/download/Con_cov.pdf>`_.

A truly zero denominator returns a finite zero. Unique coverage is calculated
when the other solution terms are supplied, including a zero result for an
empty term; it remains ``None`` when not requested.

``SufficiencyResult.membership_sum`` records the configuration membership sum.
Its optional default is ``None`` for existing manually constructed results.
The ``strength`` label is ``empty`` only when that sum is known to be zero;
otherwise a result with no cases above 0.5 is labeled
``no_cases_above_crossover``. Existing table column schemas are unchanged.

The empirical-support rules remain separate: ``is_sufficient`` still requires
at least one case above 0.5, and configuration search still applies its explicit
``min_cases`` filter (default 1). Set ``min_cases=0`` to inspect configurations
without such cases. This does not change truth-table row assignment.

Exact crossover values in truth tables
--------------------------------------

A case with membership exactly 0.5 in any crisp/fuzzy set condition has no
unambiguous truth-table corner. PyQCA warns and excludes the case from row
assignment, ``n_cases``, row ``case_ids``, and the assigned-case outcome mean.
It does not assign the case to the absent side or adjust its membership.
This rule uses exact equality: representable values immediately below or above
0.5 are still assigned normally. An outcome value of 0.5 alone does not
exclude a case.

The original data remain intact. Row consistency and PRI still use the full
case universe, including crossover cases. Frequency cutoffs apply only to
unambiguously assigned cases. A corner with no assigned cases has no observed
row and is treated as a logical remainder during minimization; a dataset with
only ambiguous cases produces an empty truth table and no positive solution.

The rule applies to FSQCA and GSQCA set conditions. GSQCA multi-value
membership-column argmax/tie handling is unchanged. See
`Dușa, QCA with R, The truth table
<https://bookdown.org/dusadrian/QCAbook/truthTable.html>`_.

Necessary-condition triviality
------------------------------

``NecessityResult.is_trivial`` is a conservative diagnostic of exact
universality: it is true only when ``is_necessary`` is true (consistency at
least 0.90) and the condition membership equals 1 in every observed case.
No tolerance is used. A condition merely above 0.5 in every case, or close to
1, is not flagged. This is not a statistical independence test or a general
test for near-triviality. Low necessity coverage measures low relevance and
is not sufficient to establish universality; high coverage does not imply
triviality. See `Dușa, QCA with R, section 5.3
<https://bookdown.org/dusadrian/QCAbook/analysisofnecessity.html>`_.

For example, ``X=[1,1,1]``, ``Y=[1,0,0]`` is flagged, while
``X=Y=[0,1]`` is not. If both X and Y are universally 1, the condition is
still universal and is flagged despite coverage 1. A zero-outcome set has
consistency 0 under the existing denominator convention, so it is not flagged;
an empty condition is likewise not a trivial necessary condition.

Engine evaluation supplies the optional ``is_universal`` membership evidence.
The existing four positional constructor arguments remain valid. For manually
constructed necessary results without that evidence, ``is_trivial`` now
returns ``None`` (unknown), rather than inferring a label from coverage.
Non-necessary results return ``False``. Dictionary and table column names are
unchanged, but consumers must allow a nullable ``is_trivial`` value for manual
results; string output explicitly reports unknown evidence. Engine-generated
results return booleans. Neither consistency nor coverage calculations are
changed by this diagnostic.

Frequency and truth-table inclusion
----------------------------------------

``build_truth_table(frequency_cutoff=1)`` and the truth table returned by
``fit(frequency_cutoff=...)`` mark a row ``include=True`` only if its observed
case count meets the frequency cutoff and its consistency and PRI meet their
respective cutoffs. Equality at each cutoff is accepted. Low-frequency rows
remain visible with their original counts, case IDs and scores, but have
``include=False``. Table exports and Markdown reports use these same flags.

``include=False`` means the row is not a positive input to minimization; it
does not by itself mean an observed negative configuration. Low-frequency
rows are still logical remainders, while sufficiently frequent rows failing
consistency or PRI remain negative. Parsimonious minimization may therefore
use low-frequency configurations as simplifying assumptions. Counts and
metrics are not recomputed from a filtered dataset.

The standalone truth-table consistency default remains 0.5, while ``fit``
uses its resolved consistency cutoff (default 0.75). To compare them directly,
pass the same consistency, PRI and frequency cutoffs to both APIs. The default
frequency cutoff of 1 preserves previous standalone behavior.

Low-membership necessity
------------------------

Necessity consistency is ``sum(min(X, Y)) / sum(Y)`` and necessity coverage
is ``sum(min(X, Y)) / sum(X)``. Positive denominators are not rounded to zero
using an absolute tolerance. For example, ``X=[2e-13, 4e-13]`` and
``Y=[3e-13, 5e-13]`` yield consistency 0.75 and coverage 1.0, while
``n_cases_in`` remains 0. Rescaling both sets preserves the ratios, subject
to floating-point representability.

Each truly zero denominator independently returns 0.0 under the existing
finite convention, not evidence of necessity. The same calculation is used
by single-literal evaluation, necessary-condition search and its result tables.
Case counts and empirical support are separate from these set-theoretic
ratios. This change does not modify PRI denominator handling or the
universality-based triviality diagnostic. See the necessity definitions in
`Dușa, QCA with R, chapter 5
<https://bookdown.org/dusadrian/QCAbook/analysisofnecessity.html>`_.
