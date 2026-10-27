"""Shared base implementation for QCA engines."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import combinations

# Type aliases (not used at runtime).
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qca._constants import (
    CROSSOVER_EPSILON,
    DONT_CARE,
    MEMBERSHIP_THRESHOLD,
    ZERO_EPSILON,
)
from qca._validation import validate_category_value
from qca.core.conditions import (
    ConditionSpec,
    condition_specs_from_schema,
    split_condition_specs,
)
from qca.core.literals import (
    MultiValueLiteral,
    SetLiteral,
)
from qca.core.results import (
    NecessityResult,
    SufficiencyResult,
    TruthTableRow,
    necessity_results_to_df,
    sufficiency_results_to_df,
    truth_table_to_df,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# QCAEngineBase
# ---------------------------------------------------------------------------


class QCAEngineBase:
    """Shared computational base for QCA engines."""

    _AUTO_CASE_ID_BASE = "__qca_case_id__"
    _DEFAULT_TRUTH_TABLE_INCLUSION_CUTOFF = 0.75

    def __init__(
        self,
        data: pd.DataFrame,
        case_id: str | None = None,
        set_conditions: Sequence[str] | None = None,
        multivalue_conditions: Sequence[str] | None = None,
        outcome: str | None = None,
        conditions: Sequence[str] | None = None,
        condition_types: dict[str, str] | None = None,
        condition_specs: Sequence[ConditionSpec] | None = None,
    ) -> None:
        # Normalize the index to zero-based integers so indices returned by
        # groupby().groups align with conjunction_membership() results.
        self.data = data.copy().reset_index(drop=True)
        self.case_id, self.generated_case_id = self._resolve_case_id(case_id)
        (
            self.condition_specs,
            self.set_conditions,
            self.multivalue_conditions,
            self.conditions,
            self.condition_types,
        ) = self._normalize_condition_schema(
            conditions=conditions,
            condition_types=condition_types,
            condition_specs=condition_specs,
            set_conditions=set_conditions,
            multivalue_conditions=multivalue_conditions,
        )
        if outcome is None:
            raise ValueError("outcome is required.")
        self.outcome = outcome
        self._materialize_multivalue_membership_columns()
        self._validate()

    def _resolve_case_id(self, case_id: str | None) -> tuple[str, bool]:
        """Resolve the case-id column, creating one when omitted."""
        if case_id is not None:
            return case_id, False

        candidate = self._AUTO_CASE_ID_BASE
        suffix = 1
        while candidate in self.data.columns:
            candidate = f"{self._AUTO_CASE_ID_BASE}_{suffix}"
            suffix += 1

        self.data[candidate] = [f"case_{i + 1}" for i in range(len(self.data))]
        return candidate, True

    @classmethod
    def from_condition_specs(
        cls,
        data: pd.DataFrame,
        outcome: str,
        condition_specs: Sequence[ConditionSpec],
        case_id: str | None = None,
    ) -> QCAEngineBase:
        """Build a model from normalized ``ConditionSpec`` objects."""
        return cls(
            data=data,
            case_id=case_id,
            outcome=outcome,
            condition_specs=condition_specs,
        )

    def _normalize_condition_schema(
        self,
        *,
        conditions: Sequence[str] | None,
        condition_types: dict[str, str] | None,
        condition_specs: Sequence[ConditionSpec] | None,
        set_conditions: Sequence[str] | None,
        multivalue_conditions: Sequence[str] | None,
    ) -> tuple[list[ConditionSpec], list[str], list[str], list[str], dict[str, str]]:
        """Normalize public and legacy condition schemas."""
        uses_specs_schema = condition_specs is not None
        uses_pyqca_schema = conditions is not None or condition_types is not None
        uses_legacy_schema = (
            set_conditions is not None or multivalue_conditions is not None
        )

        if uses_specs_schema:
            if uses_pyqca_schema or uses_legacy_schema:
                raise ValueError(
                    "condition_specs cannot be combined with conditions / "
                    "condition_types or legacy set_conditions / "
                    "multivalue_conditions."
                )
            specs = condition_specs_from_schema(condition_specs)
            return self._condition_schema_from_specs(specs)

        if uses_pyqca_schema:
            if conditions is None or condition_types is None:
                raise ValueError(
                    "conditions and condition_types must be provided together."
                )
            if uses_legacy_schema:
                raise ValueError(
                    "conditions / condition_types cannot be combined with the "
                    "legacy set_conditions / multivalue_conditions arguments."
                )
            return self._condition_schema_from_specs(
                self._normalize_pyqca_condition_schema(conditions, condition_types)
            )

        if not uses_legacy_schema:
            raise ValueError(
                "Provide conditions / condition_types or the legacy "
                "set_conditions / multivalue_conditions arguments."
            )

        set_list = list(set_conditions or [])
        multi_list = list(multivalue_conditions or [])
        self._check_duplicate_condition_names(set_list)
        self._check_duplicate_condition_names(multi_list)
        specs = [
            *[ConditionSpec(c, self._infer_set_condition_type(c)) for c in set_list],
            *[ConditionSpec(c, "multi") for c in multi_list],
        ]
        return self._condition_schema_from_specs(specs)

    @staticmethod
    def _condition_schema_from_specs(
        specs: list[ConditionSpec],
    ) -> tuple[list[ConditionSpec], list[str], list[str], list[str], dict[str, str]]:
        set_list, multi_list, condition_list, normalized_types = split_condition_specs(
            specs
        )
        return specs, set_list, multi_list, condition_list, normalized_types

    def _normalize_pyqca_condition_schema(
        self,
        conditions: Sequence[str],
        condition_types: dict[str, str],
    ) -> list[ConditionSpec]:
        condition_list = list(conditions)
        self._check_duplicate_condition_names(condition_list)

        missing_types = [c for c in condition_list if c not in condition_types]
        if missing_types:
            raise ValueError(
                "condition_types is missing entries for these conditions: "
                f"{missing_types}"
            )

        extra_types = [c for c in condition_types if c not in condition_list]
        if extra_types:
            raise ValueError(
                "condition_types contains keys not present in conditions: "
                f"{extra_types}"
            )

        specs: list[ConditionSpec] = []

        for condition in condition_list:
            condition_type = self._normalize_condition_type(
                condition, condition_types[condition]
            )
            specs.append(ConditionSpec(condition, condition_type))

        return specs

    @staticmethod
    def _normalize_condition_type(condition: str, condition_type: str) -> str:
        value = str(condition_type).strip().lower().replace("_", "-")
        if value in {"crisp", "crisp-set", "cs", "csqca"}:
            return "crisp"
        if value in {"fuzzy", "fuzzy-set", "fs", "fsqca"}:
            return "fuzzy"
        if value in {"multi", "multi-value", "multivalue", "mv", "mvqca"}:
            return "multi"
        raise ValueError(
            f"condition_types[{condition!r}] has unknown type {condition_type!r}. "
            "Available types: crisp, fuzzy, multi"
        )

    @staticmethod
    def _check_duplicate_condition_names(conditions: Sequence[str]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for condition in conditions:
            if condition in seen and condition not in duplicates:
                duplicates.append(condition)
            seen.add(condition)
        if duplicates:
            raise ValueError(f"conditions contains duplicates: {duplicates}")

    def _infer_set_condition_type(self, condition: str) -> str:
        """Infer whether a legacy set condition is crisp or fuzzy."""
        if condition not in self.data.columns:
            return "fuzzy"
        try:
            values = pd.to_numeric(self.data[condition], errors="raise").dropna()
        except (TypeError, ValueError):
            return "fuzzy"
        if not values.empty and values.astype(float).isin([0.0, 1.0]).all():
            return "crisp"
        return "fuzzy"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cases(self) -> int:
        """Return the number of cases."""
        return len(self.data)

    @property
    def n_conditions(self) -> int:
        """Return the number of conditions."""
        return len(self.set_conditions) + len(self.multivalue_conditions)

    @property
    def all_condition_names(self) -> list[str]:
        """Return all condition names in engine order."""
        return [*self.set_conditions, *self.multivalue_conditions]

    @property
    def condition_spec_map(self) -> dict[str, ConditionSpec]:
        """Return normalized condition specs keyed by condition name."""
        return {spec.name: spec for spec in self.condition_specs}

    @property
    def condition_schema(self) -> pd.DataFrame:
        """Return the normalized QCA condition schema."""
        columns = [
            "name",
            "kind",
            "role",
            "domain",
            "calibrated",
            "value_columns",
            "is_set",
            "is_multi",
        ]
        rows: list[dict[str, Any]] = []
        for spec in self.condition_specs:
            domain = spec.domain
            if domain is None:
                if spec.is_set:
                    domain = (0.0, 1.0)
                elif spec.is_multi:
                    inferred_domain = self._multivalue_domain(spec.name)
                    domain = tuple(inferred_domain) if inferred_domain else None

            calibrated = spec.calibrated
            if calibrated is None:
                calibrated = spec.is_set or bool(spec.value_columns)

            rows.append(
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "role": "set" if spec.is_set else "multi",
                    "domain": domain,
                    "calibrated": calibrated,
                    "value_columns": dict(spec.value_columns or {}),
                    "is_set": spec.is_set,
                    "is_multi": spec.is_multi,
                }
            )

        return pd.DataFrame(rows, columns=columns)

    @property
    def outcome_series(self) -> pd.Series:
        """Return the outcome series as floats."""
        return self.data[self.outcome].astype(float)

    def __repr__(self) -> str:
        engine_name = self.__class__.__name__
        return (
            f"{engine_name}("
            f"n_cases={self.n_cases}, "
            f"set_conditions={self.set_conditions}, "
            f"multivalue_conditions={self.multivalue_conditions}, "
            f"outcome={self.outcome!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def fit(
        self,
        consistency_cutoff: float | None = None,
        coverage_cutoff: float | None = None,
        minimizer: str = "standard",
        outcome_threshold: float | None = None,
        consistency_threshold: float | None = None,
        pri_cut: float = 0.0,
        frequency_cutoff: int = 1,
        directional_expectations: dict[str, Any] | None = None,
        include_remainders_in_parsimonious: bool = True,
    ):
        """Run QCA minimization and return a unified PyQCA result object.

        The low-level QMC output remains available as ``result.minimization``
        and ``result.qm_solution``. The high-level result exposes the stable
        PyQCA surface described in the README: ``truth_table``, ``solutions``,
        ``consistency``, ``coverage``, ``case_coverage``, ``to_dataframe()``,
        and ``to_markdown()``.
        """
        backend = self._normalize_minimizer_backend(minimizer)
        self._validate_optional_cutoff("coverage_cutoff", coverage_cutoff)
        self._validate_optional_cutoff("pri_cut", pri_cut)
        resolved_consistency_threshold = self._resolve_truth_table_inclusion_cutoff(
            consistency_cutoff=consistency_cutoff,
            outcome_threshold=outcome_threshold,
            consistency_threshold=consistency_threshold,
        )

        from qca.minimizers.engine import minimize_truth_table

        minimization = minimize_truth_table(
            self,
            outcome_threshold=resolved_consistency_threshold,
            consistency_threshold=resolved_consistency_threshold,
            pri_cut=pri_cut,
            frequency_cutoff=frequency_cutoff,
            directional_expectations=directional_expectations,
            include_remainders_in_parsimonious=include_remainders_in_parsimonious,
            backend=backend,
        )
        truth_table_rows = self.build_truth_table(
            outcome_threshold=resolved_consistency_threshold,
            pri_cut=pri_cut,
            frequency_cutoff=frequency_cutoff,
        )
        settings = {
            "minimizer": backend,
            "workflow": getattr(self, "workflow", getattr(self, "qca_type", None)),
            "outcome_threshold": resolved_consistency_threshold,
            "truth_table_inclusion_cutoff": resolved_consistency_threshold,
            "consistency_cutoff": resolved_consistency_threshold,
            "consistency_threshold": resolved_consistency_threshold,
            "requested_outcome_threshold": outcome_threshold,
            "requested_consistency_cutoff": consistency_cutoff,
            "pri_cut": pri_cut,
            "frequency_cutoff": frequency_cutoff,
            "coverage_cutoff": coverage_cutoff,
            "directional_expectations": directional_expectations,
            "include_remainders_in_parsimonious": include_remainders_in_parsimonious,
        }
        return self._build_fit_result(
            minimization=minimization,
            truth_table_rows=truth_table_rows,
            settings=settings,
            coverage_cutoff=coverage_cutoff,
        )

    def analyze(self, *args: Any, **kwargs: Any):
        """Alias for ``fit()`` for workflow-style high-level API usage."""
        return self.fit(*args, **kwargs)

    @staticmethod
    def _normalize_minimizer_backend(minimizer: str) -> str:
        from qca.minimizers.backends import normalize_minimizer_backend

        return normalize_minimizer_backend(minimizer)

    @staticmethod
    def _validate_optional_cutoff(name: str, value: float | None) -> None:
        if value is None:
            return
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]. Got {value}.")

    def _resolve_truth_table_inclusion_cutoff(
        self,
        *,
        consistency_cutoff: float | None,
        outcome_threshold: float | None,
        consistency_threshold: float | None,
    ) -> float:
        """Resolve the row-inclusion cutoff used for truth-table minimization."""
        for name, value in (
            ("consistency_cutoff", consistency_cutoff),
            ("outcome_threshold", outcome_threshold),
            ("consistency_threshold", consistency_threshold),
        ):
            self._validate_optional_cutoff(name, value)

        if consistency_threshold is not None:
            return float(consistency_threshold)
        if consistency_cutoff is not None:
            return float(consistency_cutoff)
        if outcome_threshold is not None:
            return float(outcome_threshold)
        return self._DEFAULT_TRUTH_TABLE_INCLUSION_CUTOFF

    def _build_fit_result(
        self,
        *,
        minimization,
        truth_table_rows: list[TruthTableRow],
        settings: dict[str, Any],
        coverage_cutoff: float | None = None,
    ):
        """Build the high-level result object from backend minimization output."""
        from qca.results.solution import QCAFitResult

        solution_metrics = {
            "complex": self._evaluate_implicant_solution(
                minimization.complex_solution,
                minimization.condition_names,
                "complex",
                coverage_cutoff=coverage_cutoff,
            ),
            "parsimonious": self._evaluate_implicant_solution(
                minimization.parsimonious_solution,
                minimization.condition_names,
                "parsimonious",
                coverage_cutoff=coverage_cutoff,
            ),
            "intermediate": self._evaluate_implicant_solution(
                minimization.intermediate_solution,
                minimization.condition_names,
                "intermediate",
                coverage_cutoff=coverage_cutoff,
            ),
        }
        selected_solution_type = self._select_solution_type(solution_metrics)
        if selected_solution_type == "none":
            selected_terms = []
        else:
            selected_terms = self._implicants_to_terms(
                getattr(minimization, f"{selected_solution_type}_solution"),
                minimization.condition_names,
            )
        case_coverage = self._build_case_coverage(selected_terms)
        return QCAFitResult(
            truth_table_rows=truth_table_rows,
            minimization=minimization,
            solution_metrics=solution_metrics,
            case_coverage=case_coverage,
            settings=settings,
            selected_solution_type=selected_solution_type,
            condition_schema=self.condition_schema,
            outcome=self.outcome,
            case_id=self.case_id,
            qca_type=getattr(self, "qca_type", self.__class__.__name__),
        )

    @staticmethod
    def _select_solution_type(
        solution_metrics: dict[str, SufficiencyResult | None],
    ) -> str:
        for solution_type in ("intermediate", "parsimonious", "complex"):
            if solution_metrics.get(solution_type) is not None:
                return solution_type
        return "none"

    def _evaluate_implicant_solution(
        self,
        implicants,
        condition_names: list[str],
        solution_type: str,
        coverage_cutoff: float | None = None,
    ) -> SufficiencyResult | None:
        terms = self._implicants_to_terms(implicants, condition_names)
        if not terms:
            return None
        x = self._solution_membership(terms)
        label = self._terms_label(terms, fallback=solution_type)
        result = self._calc_sufficiency(
            label,
            x.astype(float),
            self.outcome_series,
            solution_coverage=None,
        )
        if coverage_cutoff is not None and result.raw_coverage < coverage_cutoff:
            return None
        return result

    def _implicants_to_terms(
        self,
        implicants,
        condition_names: list[str],
    ) -> list[list[SetLiteral | MultiValueLiteral]]:
        return [
            self._implicant_to_literals(implicant, condition_names)
            for implicant in implicants
        ]

    def _implicant_to_literals(
        self,
        implicant,
        condition_names: list[str],
    ) -> list[SetLiteral | MultiValueLiteral]:
        literals: list[SetLiteral | MultiValueLiteral] = []
        for condition, value in zip(condition_names, implicant.pattern, strict=False):
            if value == DONT_CARE:
                continue
            if condition in self.set_conditions:
                literals.append(SetLiteral(condition, negated=int(value) == 0))
            elif condition in self.multivalue_conditions:
                literals.append(MultiValueLiteral(condition, value))
        return literals

    def _solution_membership(
        self,
        terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
    ) -> pd.Series:
        if not terms:
            return pd.Series(np.zeros(self.n_cases, dtype=float), index=self.data.index)
        memberships = [self._term_membership(term) for term in terms]
        return pd.concat(memberships, axis=1).max(axis=1)

    def _term_membership(
        self,
        term: Sequence[SetLiteral | MultiValueLiteral],
    ) -> pd.Series:
        if not term:
            return pd.Series(np.ones(self.n_cases, dtype=float), index=self.data.index)
        return self.conjunction_membership(term)

    @staticmethod
    def _terms_label(
        terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
        fallback: str,
    ) -> str:
        if not terms:
            return fallback
        labels: list[str] = []
        for term in terms:
            labels.append(
                " * ".join(lit.label() for lit in term) if term else "(tautology)"
            )
        return " + ".join(f"({label})" for label in labels)

    def _build_case_coverage(
        self,
        selected_terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
    ) -> pd.DataFrame:
        membership = self._solution_membership(selected_terms).astype(float)
        outcome = self.outcome_series.astype(float)
        return pd.DataFrame(
            {
                "case_id": self.data[self.case_id].astype(str),
                "outcome": outcome,
                "solution_membership": membership,
                "covered_membership": np.minimum(
                    membership.to_numpy(dtype=float),
                    outcome.to_numpy(dtype=float),
                ),
                "covered": membership.to_numpy(dtype=float) > MEMBERSHIP_THRESHOLD,
                "in_outcome": outcome.to_numpy(dtype=float) > MEMBERSHIP_THRESHOLD,
            }
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate model data and schema consistency."""
        self._check_required_columns()
        self._coerce_set_conditions()
        self._coerce_multivalue_membership_columns()
        self._coerce_outcome()
        self._check_duplicate_case_ids()
        self._check_condition_overlap()
        self._check_multivalue_missing_values()
        self._validate_declared_multivalue_domains()

    def _check_required_columns(self) -> None:
        required = [
            self.case_id,
            *self.set_conditions,
            self.outcome,
        ]
        missing = [c for c in required if c not in self.data.columns]
        for condition in self.multivalue_conditions:
            if condition in self.data.columns:
                continue
            if self._has_complete_multivalue_membership_columns(condition):
                continue
            missing.append(condition)
        if missing:
            raise ValueError(
                f"Required columns are missing from the DataFrame: {missing}\n"
                f"Available columns: {list(self.data.columns)}"
            )

    def _coerce_set_conditions(self) -> None:
        """Coerce set conditions."""
        for c in self.set_conditions:
            try:
                vals = pd.to_numeric(self.data[c], errors="raise").astype(float)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"set_condition '{c}' cannot be converted to numeric values: {e}"
                ) from e

            if not ((vals >= 0.0).all() and (vals <= 1.0).all()):
                raise ValueError(
                    f"set_condition '{c}' must be within [0, 1]. "
                    f"Observed range: [{vals.min():.4f}, {vals.max():.4f}]"
                )
            if self.condition_types.get(c) == "crisp":
                observed = vals.dropna()
                if not observed.isin([0.0, 1.0]).all():
                    raise ValueError(
                        f"crisp condition '{c}' must contain only binary "
                        "0/1 values. CSQCA requires crisp/binary values."
                    )
            # Store the converted float values back in the data.
            self.data[c] = vals

    def _materialize_multivalue_membership_columns(self) -> None:
        """Resolve all explicit aliases before writing any canonical columns."""
        protected = {self.case_id, self.outcome}
        protected.update(spec.name for spec in self.condition_specs)
        # Unmapped calibrated values still resolve by name. Do not let an
        # explicit alias overwrite or shadow a different value's input.
        implicit_columns: set[str] = set()
        for spec in self.condition_specs:
            if not spec.is_multi:
                continue
            if (
                spec.name in self.data.columns
                and spec.calibrated is not True
                and not spec.value_columns
            ):
                continue
            for value in self._multivalue_domain(spec.name):
                if value in (spec.value_columns or {}):
                    continue
                for column in self._multivalue_membership_column_candidates(
                    spec.name, value
                ):
                    implicit_columns.add(column)

        pending: dict[str, pd.Series] = {}
        for spec in self.condition_specs:
            if not spec.is_multi or not spec.value_columns:
                continue
            for value, source_column in spec.value_columns.items():
                if source_column not in self.data.columns:
                    raise ValueError(
                        f"value_columns for multi-value condition '{spec.name}' "
                        f"maps value {value!r} to missing column "
                        f"{source_column!r}."
                    )
                target_column = self._multivalue_membership_column_name(
                    spec.name, value
                )
                if target_column in protected:
                    raise ValueError(
                        f"value_columns target {target_column!r} conflicts with "
                        "an outcome, case ID, or condition column."
                    )
                if target_column in pending or target_column in implicit_columns:
                    raise ValueError(
                        f"value_columns target {target_column!r} conflicts with "
                        "another value-membership column."
                    )
                pending[target_column] = self._coerce_membership_series(
                    self.data[source_column],
                    f"{spec.name}={value}",
                ).copy()

        # Every source was read from the original data, including sources
        # shared by conditions and cycles such as A=0 <-> A=1.
        for target_column, values in pending.items():
            self.data[target_column] = values

    def _coerce_multivalue_membership_columns(self) -> None:
        """Validate calibrated multivalent fuzzy membership columns."""
        for condition in self.multivalue_conditions:
            if not self._uses_multivalue_membership_columns(condition):
                continue
            for value, column in self._multivalue_membership_column_map(
                condition
            ).items():
                self.data[column] = self._coerce_membership_series(
                    self.data[column],
                    f"{condition}={value}",
                )

    @staticmethod
    def _coerce_membership_series(series: pd.Series, label: str) -> pd.Series:
        try:
            values = pd.to_numeric(series, errors="raise").astype(float)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"membership column '{label}' cannot be converted to numeric "
                f"values: {e}"
            ) from e

        if values.isna().any():
            raise ValueError(
                f"membership column '{label}' contains missing values. "
                "Missing memberships cannot be used to construct QCA truth "
                "tables."
            )

        if not ((values >= 0.0).all() and (values <= 1.0).all()):
            raise ValueError(
                f"membership column '{label}' must be within [0, 1]. "
                f"Observed range: [{values.min():.4f}, {values.max():.4f}]"
            )
        return values

    def _coerce_outcome(self) -> None:
        """Coerce outcome."""
        try:
            y = pd.to_numeric(self.data[self.outcome], errors="raise").astype(float)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"outcome '{self.outcome}' cannot be converted to numeric values: {e}"
            ) from e

        if not ((y >= 0.0).all() and (y <= 1.0).all()):
            raise ValueError(
                f"outcome '{self.outcome}' must be within [0, 1]. "
                f"Observed range: [{y.min():.4f}, {y.max():.4f}]"
            )
        self.data[self.outcome] = y

    def _check_duplicate_case_ids(self) -> None:
        mask = self.data[self.case_id].duplicated()
        if mask.any():
            dupes = self.data[self.case_id][mask].unique().tolist()
            raise ValueError(
                f"case_id '{self.case_id}' contains duplicate values: {dupes}"
            )

    def _check_condition_overlap(self) -> None:
        overlap = set(self.set_conditions) & set(self.multivalue_conditions)
        if overlap:
            raise ValueError(
                "set_conditions and multivalue_conditions contain overlapping "
                "columns: "
                f"{sorted(overlap)}"
            )

    def _check_multivalue_missing_values(self) -> None:
        """Reject missing categorical values before logical minimization."""
        for condition in self.multivalue_conditions:
            if condition not in self.data.columns:
                continue
            missing_mask = self.data[condition].isna()
            if not missing_mask.any():
                continue
            case_ids = (
                self.data.loc[missing_mask, self.case_id].astype(str).head(5).tolist()
            )
            suffix = "" if missing_mask.sum() <= 5 else " ..."
            raise ValueError(
                f"multi-value condition '{condition}' contains missing values "
                f"for case_id(s): {case_ids}{suffix}. Missing multi-value "
                "categories are not part of the declared logical domain and "
                "would otherwise be excluded from minimization."
            )

    def _validate_declared_multivalue_domains(self) -> None:
        """Validate declared multi-value domains against observed data."""
        spec_map = self.condition_spec_map
        for condition in self.multivalue_conditions:
            if condition in self.data.columns:
                for value in self.data[condition].dropna().unique():
                    validate_category_value(value)
            spec = spec_map.get(condition)
            if spec is None or spec.domain is None:
                continue
            domain_values = list(spec.domain)
            if not domain_values:
                raise ValueError(
                    f"multi-value condition '{condition}' has an empty domain."
                )
            if condition not in self.data.columns:
                continue
            observed = self.data[condition].dropna().unique().tolist()
            outside = [value for value in observed if value not in domain_values]
            if outside:
                raise ValueError(
                    f"Observed values for multi-value condition '{condition}' "
                    f"are not present in its declared domain: {outside}. "
                    f"Declared domain: {domain_values}."
                )

    def _multivalue_domain(self, condition: str) -> list[Any]:
        """Return the declared or observed domain for a multi-value condition."""
        spec = self.condition_spec_map.get(condition)
        if spec is not None and spec.domain is not None:
            return list(spec.domain)
        if spec is not None and spec.value_columns:
            return list(spec.value_columns)
        if condition not in self.data.columns:
            return []
        return self._sorted_unique(self.data[condition])

    def _has_complete_multivalue_membership_columns(self, condition: str) -> bool:
        domain = self._multivalue_domain(condition)
        if not domain:
            return False
        return all(
            self._find_multivalue_membership_column(condition, value) is not None
            for value in domain
        )

    def _multivalue_membership_column_map(
        self,
        condition: str,
    ) -> dict[Any, str]:
        mapping: dict[Any, str] = {}
        for value in self._multivalue_domain(condition):
            column = self._find_multivalue_membership_column(condition, value)
            if column is not None:
                mapping[value] = column
        return mapping

    def _find_multivalue_membership_column(
        self,
        condition: str,
        value: Any,
    ) -> str | None:
        for candidate in self._multivalue_membership_column_candidates(
            condition, value
        ):
            if candidate in self.data.columns:
                return candidate
        return None

    @classmethod
    def _multivalue_membership_column_name(cls, condition: str, value: Any) -> str:
        return f"{condition}={cls._value_tokens(value)[0]}"

    @classmethod
    def _multivalue_membership_column_candidates(
        cls,
        condition: str,
        value: Any,
    ) -> list[str]:
        candidates: list[str] = []
        for token in cls._value_tokens(value):
            candidates.extend(
                [
                    f"{condition}={token}",
                    f"{condition}[{token}]",
                    f"{condition}__{token}",
                ]
            )
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _value_tokens(value: Any) -> list[str]:
        tokens = [str(value)]
        if isinstance(value, float) and value.is_integer():
            tokens.append(str(int(value)))
        return list(dict.fromkeys(tokens))

    # ------------------------------------------------------------------
    # Literal generation
    # ------------------------------------------------------------------

    def _all_literals(
        self, include_negations: bool = True
    ) -> list[SetLiteral | MultiValueLiteral]:
        """Generate literals for all model conditions."""
        literals: list[SetLiteral | MultiValueLiteral] = []

        for c in self.set_conditions:
            literals.append(SetLiteral(c, negated=False))
            if include_negations:
                literals.append(SetLiteral(c, negated=True))

        for c in self.multivalue_conditions:
            for v in self._multivalue_domain(c):
                literals.append(MultiValueLiteral(c, v))

        return literals

    @staticmethod
    def _sorted_unique(series: pd.Series) -> list[Any]:
        """Return unique values sorted with numeric values first."""
        unique_vals = series.dropna().unique().tolist()

        def _sort_key(v: Any) -> tuple:
            try:
                return (0, float(v), "")
            except (TypeError, ValueError):
                return (1, 0.0, str(v))

        return sorted(unique_vals, key=_sort_key)

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_contradictory(
        literals: Sequence[SetLiteral | MultiValueLiteral],
    ) -> bool:
        """Return whether a literal combination contains contradictions."""
        set_seen: dict[str, set[bool]] = {}
        mv_literals: dict[str, list[MultiValueLiteral]] = {}

        for lit in literals:
            if isinstance(lit, SetLiteral):
                negation_set = set_seen.setdefault(lit.name, set())
                negation_set.add(lit.negated)
                if len(negation_set) > 1:
                    return True
            elif isinstance(lit, MultiValueLiteral):
                same_condition = mv_literals.setdefault(lit.name, [])
                if any(lit.conflicts_with(existing) for existing in same_condition):
                    return True
                same_condition.append(lit)

        return False

    # ------------------------------------------------------------------
    # Membership calculation
    # ------------------------------------------------------------------

    def _literal_membership(
        self, literal: SetLiteral | MultiValueLiteral
    ) -> pd.Series:
        """Resolve multi-value membership using the truth-table schema policy."""
        data = self.data
        if (
            isinstance(literal, MultiValueLiteral)
            and literal.name in self.multivalue_conditions
        ):
            if self._uses_multivalue_membership_columns(literal.name):
                # Never fall back to categories for a missing calibrated value.
                data = data.drop(columns=[literal.name], errors="ignore")
            else:
                # Auxiliary columns must not shadow a categorical condition.
                data = data[[literal.name]]
        return literal.membership(data)

    def conjunction_membership(
        self,
        literals: Sequence[SetLiteral | MultiValueLiteral],
    ) -> pd.Series:
        """Compute fuzzy membership for a conjunction."""
        if not literals:
            raise ValueError("literals cannot be empty; provide at least one literal.")

        if self._is_contradictory(literals):
            labels = [lit.label() for lit in literals]
            raise ValueError(
                f"Contradictory conjunction: {labels}\n"
                "It contains both positive and negated forms of a condition "
                "(A and ~A), or multiple values of the same multi-value "
                "condition (TYPE=1 and TYPE=2)."
            )

        memberships = [self._literal_membership(lit) for lit in literals]
        return pd.concat(memberships, axis=1).min(axis=1)

    def disjunction_membership(
        self,
        terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
    ) -> pd.Series:
        """Compute fuzzy membership for a disjunction."""
        if not terms:
            raise ValueError("terms cannot be empty; provide at least one term.")

        term_memberships = [self.conjunction_membership(term) for term in terms]
        return pd.concat(term_memberships, axis=1).max(axis=1)

    # ------------------------------------------------------------------
    # Sufficiency evaluation
    # ------------------------------------------------------------------

    def evaluate_sufficiency(
        self,
        literals: Sequence[SetLiteral | MultiValueLiteral],
        solution_coverage: pd.Series | None = None,
    ) -> SufficiencyResult:
        """Evaluate sufficiency for one conjunction."""
        x = self.conjunction_membership(literals).astype(float)
        y = self.outcome_series
        label = " * ".join(lit.label() for lit in literals)
        return self._calc_sufficiency(label, x, y, solution_coverage)

    def evaluate_solution(
        self,
        terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
    ) -> SufficiencyResult:
        """Evaluate solution."""
        x = self.disjunction_membership(terms).astype(float)
        y = self.outcome_series
        label = " + ".join(
            "(" + " * ".join(lit.label() for lit in term) + ")" for term in terms
        )
        return self._calc_sufficiency(label, x, y, solution_coverage=None)

    def _calc_sufficiency(
        self,
        label: str,
        x: pd.Series,
        y: pd.Series,
        solution_coverage: pd.Series | None,
    ) -> SufficiencyResult:
        """Compute sufficiency metrics."""
        x_arr = x.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)

        n_cases_in = int((x_arr > MEMBERSHIP_THRESHOLD).sum())
        x_sum = float(x_arr.sum())
        y_sum = float(y_arr.sum())

        min_xy = np.minimum(x_arr, y_arr)
        min_xy_sum = float(min_xy.sum())

        consistency = min_xy_sum / x_sum if x_sum > 0.0 else 0.0
        raw_coverage = (
            min_xy_sum / y_sum if y_sum > 0.0 else 0.0
        )

        # Calculate unique coverage.
        unique_coverage: float | None = None
        if solution_coverage is not None:
            s_arr = solution_coverage.to_numpy(dtype=float)
            unique_num = float(
                np.sum(
                    np.maximum(
                        np.minimum(x_arr, y_arr) - np.minimum(s_arr, y_arr),
                        0.0,
                    )
                )
            )
            unique_coverage = (
                unique_num / y_sum
                if y_sum > 0.0
                else 0.0
            )

        return SufficiencyResult(
            antecedent=label,
            consistency=consistency,
            raw_coverage=raw_coverage,
            unique_coverage=unique_coverage,
            n_cases_in=n_cases_in,
            membership_sum=x_sum,
        )

    # ------------------------------------------------------------------
    # Necessity evaluation
    # ------------------------------------------------------------------

    def evaluate_necessity(
        self,
        literal: SetLiteral | MultiValueLiteral,
    ) -> NecessityResult:
        """Evaluate necessity for one literal."""
        x = self._literal_membership(literal).astype(float)
        y = self.outcome_series

        x_arr = x.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)

        y_sum = float(y_arr.sum())
        x_sum = float(x_arr.sum())
        min_xy_sum = float(np.minimum(x_arr, y_arr).sum())

        # Positive membership sums remain meaningful even below an absolute
        # tolerance. Only a truly empty denominator uses the zero convention.
        consistency = min_xy_sum / y_sum if y_sum > 0.0 else 0.0
        coverage = min_xy_sum / x_sum if x_sum > 0.0 else 0.0

        return NecessityResult(
            condition=literal.label(),
            consistency=consistency,
            coverage=coverage,
            n_cases_in=int((x_arr > MEMBERSHIP_THRESHOLD).sum()),
            is_universal=bool(x_arr.size > 0 and np.all(x_arr == 1.0)),
        )

    # ------------------------------------------------------------------
    # Unique coverage calculation
    # ------------------------------------------------------------------

    def compute_solution_coverages(
        self,
        terms: Sequence[Sequence[SetLiteral | MultiValueLiteral]],
    ) -> list[SufficiencyResult]:
        """Compute unique coverage for terms in a solution."""
        y = self.outcome_series
        results: list[SufficiencyResult] = []

        for i, term in enumerate(terms):
            other_terms = [t for j, t in enumerate(terms) if j != i]

            if other_terms:
                solution_without = self.disjunction_membership(other_terms)
            else:
                # Single-term solution: the residual solution is a zero vector.
                solution_without = pd.Series(
                    np.zeros(self.n_cases, dtype=float),
                    index=self.data.index,
                )

            x = self.conjunction_membership(term).astype(float)
            label = " * ".join(lit.label() for lit in term)
            result = self._calc_sufficiency(
                label, x, y, solution_coverage=solution_without
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Exhaustive search
    # ------------------------------------------------------------------

    def search_sufficient_configurations(
        self,
        max_depth: int = 3,
        include_negations: bool = True,
        min_consistency: float = 0.8,
        min_coverage: float = 0.1,
        min_cases: int = 1,
    ) -> pd.DataFrame:
        """Search candidate sufficient configurations."""
        all_literals = self._all_literals(include_negations=include_negations)

        # Pre-filter literals that do not meet min_cases individually.
        viable_literals = [
            lit for lit in all_literals if self._n_cases_in(lit) >= min_cases
        ]

        if not viable_literals:
            warnings.warn(
                f"No literals satisfy min_cases={min_cases}. "
                "Lower min_cases or inspect the data.",
                UserWarning,
                stacklevel=2,
            )
            return self._empty_sufficiency_df()

        all_results: list[SufficiencyResult] = []
        depth_info: list[int] = []

        for depth in range(1, max_depth + 1):
            for combo in combinations(viable_literals, depth):
                if self._is_contradictory(combo):
                    continue

                res = self.evaluate_sufficiency(combo)

                if (
                    res.consistency >= min_consistency
                    and res.raw_coverage >= min_coverage
                    and res.n_cases_in >= min_cases
                ):
                    all_results.append(res)
                    depth_info.append(depth)

        if not all_results:
            warnings.warn(
                "No sufficient conditions satisfy the requested thresholds. "
                f"(min_consistency={min_consistency}, "
                f"min_coverage={min_coverage}, "
                f"min_cases={min_cases})\n"
                "Consider lowering the thresholds or increasing max_depth.",
                UserWarning,
                stacklevel=2,
            )
            return self._empty_sufficiency_df()

        df = sufficiency_results_to_df(all_results)
        df["depth"] = depth_info

        return df.sort_values(
            ["consistency", "raw_coverage", "depth", "n_cases_in"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

    def search_necessary_conditions(
        self,
        include_negations: bool = True,
        min_consistency: float = 0.9,
        min_coverage: float = 0.1,
    ) -> pd.DataFrame:
        """Search candidate necessary conditions."""
        all_results: list[NecessityResult] = []

        for lit in self._all_literals(include_negations=include_negations):
            res = self.evaluate_necessity(lit)
            if res.consistency >= min_consistency and res.coverage >= min_coverage:
                all_results.append(res)

        if not all_results:
            warnings.warn(
                "No necessary conditions satisfy the requested thresholds. "
                f"(min_consistency={min_consistency}, "
                f"min_coverage={min_coverage})\n"
                "Consider lowering the thresholds.",
                UserWarning,
                stacklevel=2,
            )
            return necessity_results_to_df([])

        df = necessity_results_to_df(all_results)

        return df.sort_values(
            ["consistency", "coverage", "n_cases_in"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Truth-table construction
    # ------------------------------------------------------------------

    def build_truth_table(
        self,
        outcome_threshold: float = 0.5,
        pri_cut: float = 0.0,
        frequency_cutoff: int = 1,
    ) -> list[TruthTableRow]:
        """Build observed rows with frequency, consistency, and PRI inclusion.

        Low-frequency rows remain visible with include=False. Minimization
        treats them as remainders, not observed negative configurations.
        """
        self._validate_optional_cutoff("outcome_threshold", outcome_threshold)
        self._validate_optional_cutoff("pri_cut", pri_cut)
        work = self.data.copy()

        # Exactly 0.5 is not above the crossover in either polarity.
        # Exclude these cases only from empirical row assignment, retaining
        # the original data for full-universe consistency and PRI scores.
        ambiguous = work[self.set_conditions].eq(MEMBERSHIP_THRESHOLD).any(axis=1)
        if ambiguous.any():
            warnings.warn(
                f"{int(ambiguous.sum())} case(s) with a set condition exactly "
                "at 0.5 were excluded from truth-table row assignment and "
                "frequencies; consistency and PRI still use all cases.",
                UserWarning,
                stacklevel=2,
            )

        # Binarize set conditions into temporary grouping columns.
        crisp_col_map: dict[str, str] = {}
        for c in self.set_conditions:
            col = f"__crisp_{c}"
            work[col] = (work[c] > MEMBERSHIP_THRESHOLD).astype(int)
            crisp_col_map[c] = col

        multivalue_col_map: dict[str, str] = {}
        for c in self.multivalue_conditions:
            multivalue_col_map[c] = self._truth_table_multivalue_group_column(work, c)

        work = work.loc[~ambiguous]
        group_cols = [
            *crisp_col_map.values(),
            *multivalue_col_map.values(),
        ]

        rows: list[TruthTableRow] = []

        for keys, group_idx in work.groupby(
            group_cols, dropna=False, sort=True
        ).groups.items():
            # Normalize single-element groupby keys to tuples.
            if not isinstance(keys, tuple):
                keys = (keys,)

            # Build the configuration using the original column names.
            config = self._keys_to_config(keys)

            group_data = self.data.loc[group_idx]
            y_vals = group_data[self.outcome].astype(float)
            outcome_mean = float(y_vals.mean())

            # Calculate row consistency using the original calibrated scores.
            row_consist, pri_consist = self._calc_row_consistency_scores(config)

            case_ids = group_data[self.case_id].astype(str).tolist()

            rows.append(
                TruthTableRow(
                    config=config,
                    n_cases=len(group_data),
                    outcome_mean=outcome_mean,
                    outcome_raw_consist=row_consist,
                    case_ids=case_ids,
                    include=(
                        len(group_data) >= frequency_cutoff
                        and row_consist >= outcome_threshold
                        and pri_consist >= float(pri_cut)
                    ),
                    pri_consistency=pri_consist,
                )
            )

        rows.sort(key=lambda r: (-r.outcome_mean, -r.n_cases))
        return rows

    def _truth_table_multivalue_group_column(
        self,
        work: pd.DataFrame,
        condition: str,
    ) -> str:
        """Return the column used to assign multi-value cases to truth rows."""
        if not self._uses_multivalue_membership_columns(condition):
            return condition

        group_column = f"__gsqca_corner_{condition}"
        work[group_column] = self._assign_multivalue_corner(condition)
        return group_column

    def _uses_multivalue_membership_columns(self, condition: str) -> bool:
        """Return whether a multi-value condition has calibrated value sets."""
        if not self._multivalue_membership_column_map(condition):
            return False
        spec = self.condition_spec_map.get(condition)
        return (
            condition not in self.data.columns
            or (spec is not None and spec.calibrated is True)
            or (spec is not None and bool(spec.value_columns))
        )

    def _assign_multivalue_corner(self, condition: str) -> pd.Series:
        """Assign cases to the strongest value set for truth-table display."""
        domain = self._multivalue_domain(condition)
        if not domain:
            raise ValueError(
                f"multi-value condition '{condition}' needs a declared domain "
                "when it is represented by value-specific membership columns."
            )

        memberships = pd.concat(
            [
                self._literal_membership(MultiValueLiteral(condition, value))
                for value in domain
            ],
            axis=1,
        )
        memberships.columns = list(range(len(domain)))
        max_membership = memberships.max(axis=1)

        tie_mask = memberships.eq(max_membership, axis=0).sum(axis=1) > 1
        if tie_mask.any():
            warnings.warn(
                f"multi-value condition '{condition}' has {int(tie_mask.sum())} "
                "case(s) tied between value-specific membership columns. "
                "The first declared domain value is used for truth-table row "
                "assignment; consistency is still computed from fuzzy "
                "memberships over all cases.",
                UserWarning,
                stacklevel=2,
            )

        crossover_mask = (
            (max_membership - MEMBERSHIP_THRESHOLD).abs() < CROSSOVER_EPSILON
        )
        if crossover_mask.any():
            warnings.warn(
                f"multi-value condition '{condition}' has "
                f"{int(crossover_mask.sum())} case(s) whose strongest "
                "membership is exactly at the 0.5 crossover point.",
                UserWarning,
                stacklevel=2,
            )

        chosen_positions = memberships.to_numpy(dtype=float).argmax(axis=1)
        return pd.Series(
            [domain[position] for position in chosen_positions],
            index=self.data.index,
        )

    def truth_table_to_df(
        self,
        outcome_threshold: float = 0.5,
        pri_cut: float = 0.0,
    ) -> pd.DataFrame:
        """Return the truth table as a DataFrame."""
        rows = self.build_truth_table(outcome_threshold, pri_cut=pri_cut)
        return truth_table_to_df(rows)

    def _keys_to_config(self, keys: tuple[Any, ...]) -> dict[str, Any]:
        """Convert group-by keys into a configuration dictionary."""
        config: dict[str, Any] = {}
        key_iter = iter(keys)

        for orig_c in self.set_conditions:
            config[orig_c] = int(next(key_iter))

        for mv_c in self.multivalue_conditions:
            config[mv_c] = next(key_iter)

        return config

    def _calc_row_consistency(
        self,
        config: dict[str, Any],
    ) -> float:
        """Compute row-level truth-table consistency."""
        return self._calc_row_consistency_scores(config)[0]

    def _calc_row_pri_consistency(
        self,
        config: dict[str, Any],
    ) -> float:
        """Compute row-level proportional-reduction-in-inconsistency."""
        return self._calc_row_consistency_scores(config)[1]

    def _calc_row_consistency_scores(
        self,
        config: dict[str, Any],
    ) -> tuple[float, float]:
        """Compute row-level sufficiency consistency and PRI consistency."""
        config_literals = self._config_to_literals(config)

        # Fuzzy truth-table inclusion is evaluated over the full case universe,
        # not only over cases assigned to the crisp truth-table corner.
        x_vals = self.conjunction_membership(config_literals).astype(float)
        y_vals = self.outcome_series

        x_arr = x_vals.to_numpy(dtype=float)
        y_arr = y_vals.to_numpy(dtype=float)
        x_sum = float(x_arr.sum())

        if x_sum == 0.0:
            return 0.0, 0.0

        positive_overlap = float(np.minimum(x_arr, y_arr).sum())
        # Dușa, QCA with R, Eq. 6.4: intersection of X, Y, and ~Y.
        simultaneous_overlap = float(
            np.minimum(np.minimum(x_arr, y_arr), 1.0 - y_arr).sum()
        )
        consistency = positive_overlap / x_sum

        pri_denominator = x_sum - simultaneous_overlap
        if np.isclose(pri_denominator, 0.0, atol=ZERO_EPSILON):
            # No distinguishable support: retain the finite zero convention.
            pri = 0.0
        else:
            pri = (positive_overlap - simultaneous_overlap) / pri_denominator
            pri = min(1.0, max(0.0, pri))

        return float(consistency), float(pri)

    def _config_to_literals(
        self, config: dict[str, Any]
    ) -> list[SetLiteral | MultiValueLiteral]:
        """Convert a truth-table configuration into literals."""
        literals: list[SetLiteral | MultiValueLiteral] = []
        for c, v in config.items():
            if c in self.set_conditions:
                negated = int(v) == 0
                literals.append(SetLiteral(c, negated=negated))
            elif c in self.multivalue_conditions:
                literals.append(MultiValueLiteral(c, v))
        return literals

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _n_cases_in(self, lit: SetLiteral | MultiValueLiteral) -> int:
        """Return the number of cases with membership above the threshold."""
        membership = self._literal_membership(lit).to_numpy(dtype=float)
        return int((membership > MEMBERSHIP_THRESHOLD).sum())

    @staticmethod
    def _empty_sufficiency_df() -> pd.DataFrame:
        """Return an empty sufficiency-results DataFrame."""
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consistency",
                "raw_coverage",
                "unique_coverage",
                "n_cases_in",
                "is_sufficient",
                "strength",
                "depth",
            ]
        )


__all__ = ["QCAEngineBase"]
