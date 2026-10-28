Minimization backends
=====================

PyQCA separates QCA evaluation from logical minimization. Select a backend
with the ``minimizer`` argument to :meth:`qca.QCAEngineBase.fit`.

Available backends
------------------

``standard`` / ``qmc``
   Quine-McCluskey prime generation with exact minimum-cover selection.

``set_cover``
   Exact minimum-cover selection using the same solver as ``standard``.

``greedy_set_cover``
   Deterministic greedy approximation for larger candidate spaces.

``exact_set_cover``
   Exact branch-and-bound set-cover selection.

.. code-block:: python

   result = model.fit(minimizer="exact_set_cover")

Compare backends
----------------

.. code-block:: python

   from qca import benchmark_minimizers

   benchmark = benchmark_minimizers(
       model,
       minimizers=[
           "standard",
           "qmc",
           "greedy_set_cover",
           "exact_set_cover",
       ],
       outcome_threshold=0.75,
       consistency_cutoff=0.8,
   )

   print(benchmark.summary())

Complex, parsimonious, and intermediate solutions
-------------------------------------------------

The minimizer can construct complex, parsimonious, and intermediate
solutions. Intermediate solutions require directional expectations. Without
them, PyQCA reports that the intermediate solution is equivalent to the
complex solution.

With a mapping, PyQCA compares each safe remainder with observed positive
configurations. A coordinate may stay unchanged (even if contrary to the
expectation), move to the specified expected value, or vary freely when no
expectation is specified for that condition. At least one positive configuration
must support all coordinate changes together. Merely matching one expected
value is not sufficient.

Candidate remainders are also restricted to the logical region of the selected
parsimonious solution. The complex solution represents the observed positive
configurations; minimizing those configurations with the admitted remainders
therefore preserves ``complex subset intermediate subset parsimonious``.
This follows the positive-configuration comparison for easy counterfactuals
and the conservative/parsimonious pairing described in
`Dușa, QCA with R, sections 8.6–8.7
<https://bookdown.org/dusadrian/QCAbook/minimize.html>`_.
Only the selected solution pair is used; intermediate solutions for alternative,
equally optimal pairs are not enumerated. A different selected parsimonious
solution can legitimately produce a different intermediate solution.

For binary conditions, ``1`` permits changes toward presence and ``0`` toward
absence. Omitted conditions are unconstrained. ``{}`` allows all safe remainders
within the selected parsimonious region and yields its logic; ``None`` retains
the warning and complex-solution fallback. Disabling parsimonious remainders
also bounds the intermediate solution by the complex solution.
For multi-value conditions, a specified category is the preferred target:
changes to it are allowed, unchanged categories are allowed, and changes to
other categories are not. This is a categorical rule, not a numeric ordering.
Multiple preferred categories and conjunctural directional expectations are
not supported by the current mapping API. Unknown names and out-of-domain
values are still rejected before minimization.

The LF regression reproduces the published truth-table classification and
solution logic, not raw-data calibration or fuzzy fit scores. The legacy
``is_theory_consistent`` / ``filter_theory_consistent_remainders`` helpers and
``remainder_summary().theory_consistent`` report literal agreement with theory;
they do not identify the easy counterfactuals used by intermediate minimization.

Exactness and performance
-------------------------

The default ``standard`` backend, ``qmc``, ``set_cover``, and
``exact_set_cover`` select one exact cover from the generated prime implicants.
The objective is lexicographic: fewest terms, then smallest total number of
non-wildcard conditions, then smallest sorted tuple of candidate indices.
All equally optimal solutions are not enumerated. Complex, parsimonious,
and intermediate solutions use the same covering objective.

Exact cover selection can require exponential time as candidate spaces grow.
There is no automatic fallback to an approximation. Select
``greedy_set_cover`` explicitly when that tradeoff is acceptable; its term
count and complexity are not guaranteed to be minimal. The shared prime
implicant generation can also be expensive, including for the greedy backend.

The exact selector raises ``ValueError`` if any requested target cannot be
covered, including a nonempty target with no candidates. It returns an empty
list for an empty target. This replaces the previous partial-cover behavior.
