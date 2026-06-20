"""Backend registry for PyQCA minimizers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimizerBackend:
    """Description of one minimization backend."""

    name: str
    family: str
    description: str
    exact: bool


_BACKENDS: dict[str, MinimizerBackend] = {
    "standard": MinimizerBackend(
        name="standard",
        family="boolean",
        description="Default standard Boolean minimization backend.",
        exact=False,
    ),
    "qmc": MinimizerBackend(
        name="qmc",
        family="quine_mccluskey",
        description="Quine-McCluskey-style prime implicant minimizer.",
        exact=False,
    ),
    "set_cover": MinimizerBackend(
        name="set_cover",
        family="set_cover",
        description="Set-covering backend using exact cover selection.",
        exact=True,
    ),
    "greedy_set_cover": MinimizerBackend(
        name="greedy_set_cover",
        family="set_cover",
        description="Greedy set-cover approximation backend.",
        exact=False,
    ),
    "exact_set_cover": MinimizerBackend(
        name="exact_set_cover",
        family="set_cover",
        description="Exact branch-and-bound set-cover backend.",
        exact=True,
    ),
}

_ALIASES = {
    "default": "standard",
    "standard": "standard",
    "boolean": "standard",
    "standard-boolean": "standard",
    "qmc": "qmc",
    "qm": "qmc",
    "quine-mccluskey": "qmc",
    "quine mccluskey": "qmc",
    "set-cover": "set_cover",
    "set-covering": "set_cover",
    "set_cover": "set_cover",
    "set_covering": "set_cover",
    "greedy": "greedy_set_cover",
    "greedy-set-cover": "greedy_set_cover",
    "greedy-set-covering": "greedy_set_cover",
    "greedy_set_cover": "greedy_set_cover",
    "greedy_set_covering": "greedy_set_cover",
    "exact": "exact_set_cover",
    "exact-set-cover": "exact_set_cover",
    "exact-set-covering": "exact_set_cover",
    "exact_set_cover": "exact_set_cover",
    "exact_set_covering": "exact_set_cover",
}


def available_minimizers() -> tuple[str, ...]:
    """Return canonical backend names accepted by ``fit(minimizer=...)``."""
    return tuple(_BACKENDS)


def get_minimizer_backend(name: str) -> MinimizerBackend:
    """Return backend metadata by canonical or alias name."""
    return _BACKENDS[normalize_minimizer_backend(name)]


def normalize_minimizer_backend(name: str) -> str:
    """Normalize minimizer backend names and aliases."""
    value = str(name).strip().lower().replace("_", "-")
    normalized = _ALIASES.get(value)
    if normalized is None:
        available = ", ".join(available_minimizers())
        raise ValueError(
            f"Unsupported minimizer {name!r}. Available minimizers: {available}"
        )
    return normalized


__all__ = [
    "MinimizerBackend",
    "available_minimizers",
    "get_minimizer_backend",
    "normalize_minimizer_backend",
]
