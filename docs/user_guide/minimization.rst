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
