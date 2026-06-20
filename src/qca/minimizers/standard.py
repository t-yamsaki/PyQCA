"""Default Boolean minimization backend.

For now, PyQCA's standard backend is the Quine-McCluskey implementation.
"""

from qca.minimizers.qmc import QCAMinimizer, QMSolution, minimize_truth_table

__all__ = ["QCAMinimizer", "QMSolution", "minimize_truth_table"]
