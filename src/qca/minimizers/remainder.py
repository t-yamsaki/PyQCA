"""Logical-remainder utilities for QCA minimization."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

from qca._constants import (
    DONT_CARE,
)
from qca._types import (
    ConditionDomains,
    DirectionalExpectations,
    MintermList,
    TruthTableRowProtocol,
)

# ---------------------------------------------------------------------------
# 全論理空間の生成
# ---------------------------------------------------------------------------


def generate_all_minterms(
    condition_names: list[str],
    condition_domains: ConditionDomains,
) -> list[tuple[Any, ...]]:
    """Enumerate the full logical space."""
    if not condition_names:
        return [()]

    domains = [condition_domains[c] for c in condition_names]
    return list(product(*domains))


# ---------------------------------------------------------------------------
# 真理表行のパターン変換
# ---------------------------------------------------------------------------


def row_to_pattern(
    config: dict[str, Any],
    condition_names: list[str],
    is_multivalue_flags: list[bool],
) -> tuple[Any, ...] | None:
    """Row to pattern."""
    pattern: list[Any] = []
    for c, is_mv in zip(condition_names, is_multivalue_flags, strict=False):
        if c not in config:
            return None
        val = config[c]
        if not is_mv:
            val = int(val)
        pattern.append(val)
    return tuple(pattern)


# ---------------------------------------------------------------------------
# 真理表行の分類
# ---------------------------------------------------------------------------


def classify_truth_table_rows(
    truth_table_rows: list[TruthTableRowProtocol],
    all_patterns: list[tuple[Any, ...]],
    pattern_to_idx: dict[tuple[Any, ...], int],
    condition_names: list[str],
    is_multivalue_flags: list[bool],
    outcome_threshold: float,
    consistency_threshold: float,
    frequency_cutoff: int = 1,
) -> tuple[MintermList, MintermList, MintermList, MintermList]:
    """Classify truth-table rows as positive, negative, remainder, or contradiction."""
    observed_patterns: set[tuple[Any, ...]] = set()
    positive_set: set[int] = set()
    negative_set: set[int] = set()
    contradiction_set: set[int] = set()

    for row in truth_table_rows:
        if getattr(row, "n_cases", 1) < frequency_cutoff:
            continue

        pat = row_to_pattern(row.config, condition_names, is_multivalue_flags)
        if pat is None:
            warnings.warn(
                f"真理表行の config {row.config} を既知のパターンに変換できません。"
                "スキップします。condition_names と is_multivalue_flags を確認してください。",
                UserWarning,
                stacklevel=3,
            )
            continue

        idx = pattern_to_idx.get(pat)
        if idx is None:
            warnings.warn(
                f"パターン {pat} が全論理空間に存在しません。"
                "condition_domains の設定を確認してください。",
                UserWarning,
                stacklevel=3,
            )
            continue

        observed_patterns.add(pat)

        is_positive = (
            row.outcome_mean >= outcome_threshold
            and row.outcome_raw_consist >= consistency_threshold
        )

        # 矛盾チェック: 同一インデックスが既に反対の分類に存在するか
        if (
            is_positive
            and idx in negative_set
            or not is_positive
            and idx in positive_set
        ):
            contradiction_set.add(idx)

        if is_positive:
            positive_set.add(idx)
        else:
            negative_set.add(idx)

    # 矛盾行を正・負セットから除去
    positive_indices = sorted(positive_set - contradiction_set)
    negative_indices = sorted(negative_set - contradiction_set)
    contradiction_indices = sorted(contradiction_set)

    # 残余行: 全論理空間から観察行を除いたもの
    remainder_indices = [
        i for i, pat in enumerate(all_patterns) if pat not in observed_patterns
    ]

    if contradiction_indices:
        warnings.warn(
            f"{len(contradiction_indices)} 件の矛盾行が検出され、"
            "論理最小化から除外されました。"
            "キャリブレーションの見直しを検討してください。"
            f"(Ragin 2008, p.128)",
            UserWarning,
            stacklevel=2,
        )

    return (
        positive_indices,
        negative_indices,
        remainder_indices,
        contradiction_indices,
    )


# ---------------------------------------------------------------------------
# 残余行の方向性フィルタ
# ---------------------------------------------------------------------------


def is_theory_consistent(
    pattern: tuple[Any, ...],
    condition_names: list[str],
    directional_expectations: DirectionalExpectations,
) -> bool:
    """Is theory consistent."""
    for name, val in zip(condition_names, pattern, strict=False):
        if val == DONT_CARE:
            continue
        if name not in directional_expectations:
            continue
        expected = directional_expectations[name]
        if val != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# 安全な残余行の抽出
# ---------------------------------------------------------------------------


def filter_safe_remainders(
    remainder_indices: MintermList,
    negative_indices: MintermList,
    contradiction_indices: MintermList,
) -> MintermList:
    """Filter safe remainders."""
    excluded = set(negative_indices) | set(contradiction_indices)
    return [i for i in remainder_indices if i not in excluded]


def filter_theory_consistent_remainders(
    remainder_indices: MintermList,
    negative_indices: MintermList,
    contradiction_indices: MintermList,
    all_patterns: list[tuple[Any, ...]],
    condition_names: list[str],
    directional_expectations: DirectionalExpectations,
) -> MintermList:
    """Filter theory consistent remainders."""
    safe = filter_safe_remainders(
        remainder_indices, negative_indices, contradiction_indices
    )
    return [
        i
        for i in safe
        if is_theory_consistent(
            all_patterns[i], condition_names, directional_expectations
        )
    ]
