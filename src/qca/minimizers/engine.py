"""Engine module."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

from qca._constants import (
    ROW_CONTRADICTION,
    ROW_NEGATIVE,
    ROW_POSITIVE,
    ROW_REMAINDER,
)
from qca._types import (
    ConditionDomains,
    DirectionalExpectations,
    IsMultiValueMap,
    MintermList,
    TruthTableRowProtocol,
)
from qca.minimizers.algorithms import (
    build_coverage_table,
    quine_mccluskey,
    select_essential_prime_implicants,
)
from qca.minimizers.implicant import Implicant, QMSolution
from qca.minimizers.remainder import (
    classify_truth_table_rows,
    filter_safe_remainders,
    filter_theory_consistent_remainders,
    generate_all_minterms,
    is_theory_consistent,
)

if TYPE_CHECKING:
    from qca.engines.base import QCAEngineBase


# ---------------------------------------------------------------------------
# QCAMinimizer
# ---------------------------------------------------------------------------


class QCAMinimizer:
    """Logical minimization engine for QCA truth tables."""

    def __init__(
        self,
        condition_names: list[str],
        is_multivalue: IsMultiValueMap,
        condition_domains: ConditionDomains,
        backend_name: str = "qmc",
    ) -> None:
        self.backend_name = backend_name
        self.condition_names = condition_names
        self.is_multivalue_flags: list[bool] = [
            is_multivalue.get(c, False) for c in condition_names
        ]
        self.condition_domains = condition_domains

        # 全論理空間を事前生成してキャッシュする
        self._all_patterns: list[tuple[Any, ...]] = generate_all_minterms(
            condition_names, condition_domains
        )
        # パターン → インデックスの逆引き辞書
        self._pattern_to_idx: dict[tuple[Any, ...], int] = {
            pat: i for i, pat in enumerate(self._all_patterns)
        }

    @property
    def n_minterms(self) -> int:
        """N minterms."""
        return len(self._all_patterns)

    def __repr__(self) -> str:
        return (
            f"QCAMinimizer("
            f"conditions={self.condition_names}, "
            f"n_minterms={self.n_minterms}"
            f")"
        )

    # ------------------------------------------------------------------
    # メイン最小化
    # ------------------------------------------------------------------

    def minimize(
        self,
        truth_table_rows: list[TruthTableRowProtocol],
        outcome_threshold: float = 0.75,
        consistency_threshold: float = 0.75,
        frequency_cutoff: int = 1,
        directional_expectations: DirectionalExpectations | None = None,
        include_remainders_in_parsimonious: bool = True,
    ) -> QMSolution:
        """Derive complex, parsimonious, and intermediate solutions."""
        # Step 1: 真理表行の分類
        (
            positive_indices,
            negative_indices,
            remainder_indices,
            contradiction_indices,
        ) = classify_truth_table_rows(
            truth_table_rows=truth_table_rows,
            all_patterns=self._all_patterns,
            pattern_to_idx=self._pattern_to_idx,
            condition_names=self.condition_names,
            is_multivalue_flags=self.is_multivalue_flags,
            outcome_threshold=outcome_threshold,
            consistency_threshold=consistency_threshold,
            frequency_cutoff=frequency_cutoff,
        )

        solution = QMSolution(
            backend=self.backend_name,
            condition_names=self.condition_names,
            n_positive=len(positive_indices),
            n_remainder=len(remainder_indices),
            n_contradiction=len(contradiction_indices),
        )

        if not positive_indices:
            warnings.warn(
                "ポジティブ行が見つかりませんでした。"
                f"outcome_threshold ({outcome_threshold}) または "
                f"consistency_threshold ({consistency_threshold}) を "
                "下げることを検討してください。",
                UserWarning,
                stacklevel=2,
            )
            return solution

        # Step 2: 複雑解
        solution = self._build_complex_solution(solution, positive_indices)

        # Step 3: 倹約解
        solution = self._build_parsimonious_solution(
            solution,
            positive_indices,
            remainder_indices,
            negative_indices,
            contradiction_indices,
            include_remainders_in_parsimonious,
        )

        # Step 4: 中間解
        solution = self._build_intermediate_solution(
            solution,
            positive_indices,
            remainder_indices,
            negative_indices,
            contradiction_indices,
            directional_expectations,
        )

        # カバレッジ表の構築（倹約解のプライム含意節を使用）
        solution.coverage_table = build_coverage_table(
            solution.prime_implicants_parsimonious,
            positive_indices,
            self.condition_names,
        )

        return solution

    # ------------------------------------------------------------------
    # 三種の解の構築（プライベートメソッド）
    # ------------------------------------------------------------------

    def _build_complex_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
    ) -> QMSolution:
        """Build complex solution."""
        pi_complex = quine_mccluskey(
            minterms=positive_indices,
            all_patterns=self._all_patterns,
            is_multivalue=self.is_multivalue_flags,
            dont_care_indices=None,
        )
        solution.prime_implicants_complex = pi_complex
        solution.complex_solution = self._select_prime_implicants(
            pi_complex, positive_indices
        )
        return solution

    def _build_parsimonious_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
        remainder_indices: MintermList,
        negative_indices: MintermList,
        contradiction_indices: MintermList,
        include_remainders: bool,
    ) -> QMSolution:
        """Build parsimonious solution."""
        if not include_remainders:
            # 残余行を使用しない場合は複雑解と同一
            solution.prime_implicants_parsimonious = solution.prime_implicants_complex
            solution.parsimonious_solution = solution.complex_solution
            return solution

        safe_remainders = filter_safe_remainders(
            remainder_indices, negative_indices, contradiction_indices
        )
        pi_parsimonious = quine_mccluskey(
            minterms=positive_indices,
            all_patterns=self._all_patterns,
            is_multivalue=self.is_multivalue_flags,
            dont_care_indices=safe_remainders,
        )
        solution.prime_implicants_parsimonious = pi_parsimonious
        solution.parsimonious_solution = self._select_prime_implicants(
            pi_parsimonious, positive_indices
        )
        return solution

    def _build_intermediate_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
        remainder_indices: MintermList,
        negative_indices: MintermList,
        contradiction_indices: MintermList,
        directional_expectations: DirectionalExpectations | None,
    ) -> QMSolution:
        """Build intermediate solution."""
        if directional_expectations is None:
            warnings.warn(
                "directional_expectations が指定されていません。"
                "中間解は複雑解と同一になります。"
                "理論的期待方向を指定することで中間解を得られます。"
                "(Ragin 2008, p.160-168)",
                UserWarning,
                stacklevel=3,
            )
            solution.prime_implicants_intermediate = solution.prime_implicants_complex
            solution.intermediate_solution = solution.complex_solution
            return solution

        theory_remainders = filter_theory_consistent_remainders(
            remainder_indices=remainder_indices,
            negative_indices=negative_indices,
            contradiction_indices=contradiction_indices,
            all_patterns=self._all_patterns,
            condition_names=self.condition_names,
            directional_expectations=directional_expectations,
        )
        pi_intermediate = quine_mccluskey(
            minterms=positive_indices,
            all_patterns=self._all_patterns,
            is_multivalue=self.is_multivalue_flags,
            dont_care_indices=theory_remainders,
        )
        solution.prime_implicants_intermediate = pi_intermediate
        solution.intermediate_solution = self._select_prime_implicants(
            pi_intermediate, positive_indices
        )
        return solution

    def _select_prime_implicants(
        self,
        prime_implicants: list[Implicant],
        positive_indices: MintermList,
    ) -> list[Implicant]:
        """Select a cover from prime implicants.

        Subclasses override this hook to provide alternative set-covering
        strategies while retaining the same truth-table and remainder logic.
        """
        return select_essential_prime_implicants(
            prime_implicants,
            positive_indices,
        )

    # ------------------------------------------------------------------
    # 診断用メソッド
    # ------------------------------------------------------------------

    def classify_rows(
        self,
        truth_table_rows: list[TruthTableRowProtocol],
        outcome_threshold: float = 0.75,
        consistency_threshold: float = 0.75,
        frequency_cutoff: int = 1,
    ) -> dict[str, MintermList]:
        """Classify rows."""
        pos, neg, rem, con = classify_truth_table_rows(
            truth_table_rows=truth_table_rows,
            all_patterns=self._all_patterns,
            pattern_to_idx=self._pattern_to_idx,
            condition_names=self.condition_names,
            is_multivalue_flags=self.is_multivalue_flags,
            outcome_threshold=outcome_threshold,
            consistency_threshold=consistency_threshold,
            frequency_cutoff=frequency_cutoff,
        )
        return {
            ROW_POSITIVE: pos,
            ROW_NEGATIVE: neg,
            ROW_REMAINDER: rem,
            ROW_CONTRADICTION: con,
        }

    def remainder_summary(
        self,
        truth_table_rows: list[TruthTableRowProtocol],
        outcome_threshold: float = 0.75,
        consistency_threshold: float = 0.75,
        frequency_cutoff: int = 1,
        directional_expectations: DirectionalExpectations | None = None,
    ) -> pd.DataFrame:
        """Return diagnostic remainder classification results."""
        _, _, remainder_indices, _ = classify_truth_table_rows(
            truth_table_rows=truth_table_rows,
            all_patterns=self._all_patterns,
            pattern_to_idx=self._pattern_to_idx,
            condition_names=self.condition_names,
            is_multivalue_flags=self.is_multivalue_flags,
            outcome_threshold=outcome_threshold,
            consistency_threshold=consistency_threshold,
            frequency_cutoff=frequency_cutoff,
        )

        records: list[dict[str, Any]] = []
        for idx in remainder_indices:
            pat = self._all_patterns[idx]
            row: dict[str, Any] = {
                c: v for c, v in zip(self.condition_names, pat, strict=False)
            }
            row["classification"] = ROW_REMAINDER
            row["theory_consistent"] = (
                is_theory_consistent(
                    pat, self.condition_names, directional_expectations
                )
                if directional_expectations is not None
                else None
            )
            records.append(row)

        if not records:
            return pd.DataFrame(
                columns=[
                    *self.condition_names,
                    "classification",
                    "theory_consistent",
                ]
            )

        return (
            pd.DataFrame(records)
            .sort_values("theory_consistent", ascending=False, na_position="last")
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# ヘルパー関数: MixedQCA からの直接実行
# ---------------------------------------------------------------------------


def minimize_truth_table(
    qca: QCAEngineBase,
    outcome_threshold: float = 0.75,
    consistency_threshold: float = 0.75,
    frequency_cutoff: int = 1,
    directional_expectations: DirectionalExpectations | None = None,
    include_remainders_in_parsimonious: bool = True,
    backend: str = "qmc",
) -> QMSolution:
    """Run QCA minimization directly from a model instance."""
    # 条件ドメインの自動構築
    condition_domains: ConditionDomains = {}
    is_multivalue: IsMultiValueMap = {}

    for c in qca.set_conditions:
        condition_domains[c] = [0, 1]
        is_multivalue[c] = False

    for c in qca.multivalue_conditions:
        unique_vals = qca._sorted_unique(qca.data[c])
        condition_domains[c] = unique_vals
        is_multivalue[c] = True

    condition_names = qca.all_condition_names

    from qca.minimizers.backends import normalize_minimizer_backend

    normalized_backend = normalize_minimizer_backend(backend)
    if normalized_backend in {"standard", "qmc"}:
        minimizer = QCAMinimizer(
            condition_names=condition_names,
            is_multivalue=is_multivalue,
            condition_domains=condition_domains,
            backend_name=normalized_backend,
        )
    else:
        from qca.minimizers.set_cover import SetCoverMinimizer

        cover_method = "greedy" if normalized_backend == "greedy_set_cover" else "exact"
        minimizer = SetCoverMinimizer(
            condition_names=condition_names,
            is_multivalue=is_multivalue,
            condition_domains=condition_domains,
            cover_method=cover_method,
            backend_name=normalized_backend,
        )

    truth_table_rows = qca.build_truth_table(outcome_threshold=outcome_threshold)

    return minimizer.minimize(
        truth_table_rows=truth_table_rows,
        outcome_threshold=outcome_threshold,
        consistency_threshold=consistency_threshold,
        frequency_cutoff=frequency_cutoff,
        directional_expectations=directional_expectations,
        include_remainders_in_parsimonious=include_remainders_in_parsimonious,
    )
