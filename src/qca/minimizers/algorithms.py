"""Algorithms module."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import DONT_CARE
from qca.minimizers.implicant import Implicant

# ---------------------------------------------------------------------------
# マージ可否判定
# ---------------------------------------------------------------------------


def can_merge(
    p1: tuple[Any, ...],
    p2: tuple[Any, ...],
    is_multivalue: list[bool],
) -> tuple[bool, tuple[Any, ...] | None]:
    """Return whether two patterns can be merged by the QM algorithm."""
    if len(p1) != len(p2):
        return False, None

    diff_positions: list[int] = []

    for i, (v1, v2) in enumerate(zip(p1, p2, strict=False)):
        # ワイルドカード状態の不一致 → マージ不可
        if (v1 == DONT_CARE) != (v2 == DONT_CARE):
            return False, None
        if v1 != v2:
            diff_positions.append(i)

    # 差異が厳密に 1 箇所のみ、かつ二値条件の位置
    if len(diff_positions) == 1:
        pos = diff_positions[0]
        if not is_multivalue[pos]:
            merged = list(p1)
            merged[pos] = DONT_CARE
            return True, tuple(merged)

    return False, None


# ---------------------------------------------------------------------------
# Quine-McCluskey 法の中核
# ---------------------------------------------------------------------------


def quine_mccluskey(
    minterms: list[int],
    all_patterns: list[tuple[Any, ...]],
    is_multivalue: list[bool],
    dont_care_indices: list[int] | None = None,
) -> list[Implicant]:
    """Quine mccluskey."""
    if not minterms:
        return []

    dont_care_set: frozenset[int] = frozenset(dont_care_indices or [])
    target_set: frozenset[int] = frozenset(minterms)

    # 初期インプリカントの生成
    # キー: カバーするミンタームインデックスの frozenset
    # 値: パターンタプル
    initial_indices = sorted(target_set | dont_care_set)
    current: dict[frozenset[int], tuple[Any, ...]] = {
        frozenset([i]): all_patterns[i] for i in initial_indices
    }

    all_prime_candidates: list[Implicant] = []

    while current:
        next_level: dict[frozenset[int], tuple[Any, ...]] = {}
        merged_keys: set[frozenset[int]] = set()

        items = list(current.items())
        n = len(items)

        for i in range(n):
            for j in range(i + 1, n):
                cov1, pat1 = items[i]
                cov2, pat2 = items[j]

                ok, merged_pat = can_merge(pat1, pat2, is_multivalue)
                if ok and merged_pat is not None:
                    new_cov = cov1 | cov2
                    # 同じカバレッジのエントリが既に存在する場合は上書き
                    next_level[new_cov] = merged_pat
                    merged_keys.add(cov1)
                    merged_keys.add(cov2)

        # マージされなかったインプリカントがプライム含意節候補
        for cov, pat in items:
            if cov not in merged_keys:
                # ポジティブ行を少なくとも 1 つカバーしているか確認
                positive_covered = cov & target_set
                if positive_covered:
                    all_prime_candidates.append(
                        Implicant(
                            pattern=pat,
                            covered=frozenset(positive_covered),
                            is_prime=True,
                        )
                    )

        current = next_level

    return deduplicate_implicants(all_prime_candidates)


# ---------------------------------------------------------------------------
# 重複除去
# ---------------------------------------------------------------------------


def deduplicate_implicants(
    implicants: list[Implicant],
) -> list[Implicant]:
    """Merge implicants with identical patterns."""
    seen: dict[tuple[Any, ...], frozenset[int]] = {}
    for imp in implicants:
        seen[imp.pattern] = seen.get(imp.pattern, frozenset()) | imp.covered

    return [
        Implicant(pattern=pat, covered=cov, is_prime=True) for pat, cov in seen.items()
    ]


# ---------------------------------------------------------------------------
# 必須プライム含意節の選択
# ---------------------------------------------------------------------------


def select_essential_prime_implicants(
    prime_implicants: list[Implicant],
    target_minterms: list[int],
) -> list[Implicant]:
    """Select essential prime implicants for a minimum cover."""
    if not prime_implicants or not target_minterms:
        return []

    target_set = set(target_minterms)
    selected: list[Implicant] = []
    selected_set: set[int] = set()  # 選択済みのインデックス（重複防止）
    covered: set[int] = set()

    # Step 1: 必須プライム含意節の同定
    for m in target_minterms:
        covering = [
            (i, imp) for i, imp in enumerate(prime_implicants) if m in imp.covered
        ]
        if len(covering) == 1:
            idx, imp = covering[0]
            if idx not in selected_set:
                selected.append(imp)
                selected_set.add(idx)
                covered |= imp.covered

    # Step 2: greedy 補完
    remaining = target_set - covered
    while remaining:
        # 未カバーミンタームを最も多くカバーし、かつ complexity が最小のものを選択
        best_idx, best_imp = max(
            enumerate(prime_implicants),
            key=lambda t: (
                len(t[1].covered & remaining),
                -t[1].complexity(),  # complexity が小さいほど優先
            ),
        )

        if not (best_imp.covered & remaining):
            # カバー不可能なミンタームが残存（矛盾行等の異常ケース）
            warnings.warn(
                f"以下のミンタームをカバーできません: {sorted(remaining)}。"
                "真理表に矛盾行が存在しないか確認してください。",
                UserWarning,
                stacklevel=3,
            )
            break

        if best_idx not in selected_set:
            selected.append(best_imp)
            selected_set.add(best_idx)
        covered |= best_imp.covered
        remaining = target_set - covered

    # 元のリストの順序に従いソート
    order = {id(imp): i for i, imp in enumerate(prime_implicants)}
    selected.sort(key=lambda imp: order.get(id(imp), 0))

    return selected


# ---------------------------------------------------------------------------
# カバレッジ表の構築
# ---------------------------------------------------------------------------


def build_coverage_table(
    prime_implicants: list[Implicant],
    target_minterms: list[int],
    condition_names: list[str],
) -> pd.DataFrame:
    """Build the implicant-by-minterm coverage table."""
    if not prime_implicants or not target_minterms:
        return pd.DataFrame()

    labels = [imp.label(condition_names) for imp in prime_implicants]
    columns = [f"m{i}" for i in target_minterms]

    data = np.zeros((len(prime_implicants), len(target_minterms)), dtype=int)
    for i, imp in enumerate(prime_implicants):
        for j, m in enumerate(target_minterms):
            if m in imp.covered:
                data[i, j] = 1

    df = pd.DataFrame(data, index=labels, columns=columns)
    df.index.name = "implicant"
    df["complexity"] = [imp.complexity() for imp in prime_implicants]
    df["n_covered"] = [len(imp.covered) for imp in prime_implicants]

    return df
