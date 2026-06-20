"""Optional Matplotlib visualizations for PyQCA.

Matplotlib is intentionally optional. Importing this module does not require it;
calling a plotting function does. Install with ``pyqca[viz]`` to enable plots.
"""

from qca.viz._helpers import (
    add_crossover_lines,
    add_stat_box,
    annotate_cases,
    apply_grid_style,
    consistency_color,
    consistency_legend_patches,
    save_or_show,
    scatter_point_color,
)
from qca.viz.bar_plots import plot_necessity_bar, plot_sufficiency_bar
from qca.viz.configuration_chart import configuration_matrix, plot_configuration_chart
from qca.viz.coverage import draw_unique_coverage, plot_unique_coverage
from qca.viz.dashboard import _parse_best_literals, plot_dashboard
from qca.viz.distribution import (
    draw_membership_distribution,
    plot_membership_distribution,
)
from qca.viz.truth_table import (
    _build_condition_matrix,
    draw_truth_table,
    plot_truth_table,
)
from qca.viz.xy_plot import plot_xy

__all__ = [
    "consistency_color",
    "scatter_point_color",
    "save_or_show",
    "apply_grid_style",
    "add_crossover_lines",
    "add_stat_box",
    "annotate_cases",
    "consistency_legend_patches",
    "plot_xy",
    "plot_sufficiency_bar",
    "plot_necessity_bar",
    "configuration_matrix",
    "plot_configuration_chart",
    "_build_condition_matrix",
    "draw_truth_table",
    "plot_truth_table",
    "draw_unique_coverage",
    "plot_unique_coverage",
    "draw_membership_distribution",
    "plot_membership_distribution",
    "_parse_best_literals",
    "plot_dashboard",
]
