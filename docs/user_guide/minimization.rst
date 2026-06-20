Minimization backends
=====================

PyQCA separates QCA evaluation from logical minimization. Select a backend
with the ``minimizer`` argument to :meth:`qca.QCAEngineBase.fit`.

Available backends
------------------

``standard`` / ``qmc``
   Quine-McCluskey-style Boolean minimization.

``set_cover``
   Set-covering minimization through the default selector.

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
