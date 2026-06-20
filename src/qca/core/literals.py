"""Literals module."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca._constants import (
    CROSSOVER_EPSILON,
    FULL_MEMBERSHIP,
    MEMBERSHIP_THRESHOLD,
    NO_MEMBERSHIP,
)

# ---------------------------------------------------------------------------
# SetLiteral
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetLiteral:
    """Literal for a crisp-set or fuzzy-set condition."""

    name: str
    negated: bool = False

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Return a human-readable label."""
        return f"~{self.name}" if self.negated else self.name

    def __repr__(self) -> str:
        return f"SetLiteral({self.label()!r})"

    # ------------------------------------------------------------------
    # メンバーシップ計算
    # ------------------------------------------------------------------

    def membership(self, df: pd.DataFrame) -> pd.Series:
        """Return membership scores for this literal."""
        if self.name not in df.columns:
            raise KeyError(
                f"SetLiteral '{self.name}' が DataFrame に存在しません。"
                f"利用可能なカラム: {list(df.columns)}"
            )

        s = pd.to_numeric(df[self.name], errors="raise").astype(float)

        # 値域チェック: [0.0, 1.0]
        if not ((s >= NO_MEMBERSHIP).all() and (s <= FULL_MEMBERSHIP).all()):
            raise ValueError(
                f"SetLiteral '{self.name}' の値は [0, 1] の範囲内である必要があります。"
                f"検出された範囲: [{s.min():.4f}, {s.max():.4f}]"
            )

        # クロスオーバーポイントの警告
        # Ragin (2008) p.30: 0.5 のケースは分析的に曖昧
        at_crossover = (s - MEMBERSHIP_THRESHOLD).abs() < CROSSOVER_EPSILON
        if at_crossover.any():
            n_at = int(at_crossover.sum())
            warnings.warn(
                f"SetLiteral '{self.name}' に {n_at} 件のケースが "
                f"クロスオーバーポイント (0.5) に存在します。"
                f"Ragin (2008) p.30 はメンバーシップの曖昧さを避けるため "
                f"0.5 への配置を推奨していません。キャリブレーションを見直してください。",
                UserWarning,
                stacklevel=2,
            )

        return (FULL_MEMBERSHIP - s) if self.negated else s

    # ------------------------------------------------------------------
    # 集合演算ユーティリティ
    # ------------------------------------------------------------------

    def negate(self) -> SetLiteral:
        """Return the negated literal."""
        return SetLiteral(self.name, not self.negated)

    def is_negation_of(self, other: object) -> bool:
        """Return whether another literal is the logical negation of this one."""
        if not isinstance(other, SetLiteral):
            return False
        return self.name == other.name and self.negated != other.negated


# ---------------------------------------------------------------------------
# MultiValueLiteral
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiValueLiteral:
    """Literal for a multi-value QCA condition."""

    name: str
    value: Any

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Return a human-readable label."""
        return f"{self.name}={self.value}"

    def __repr__(self) -> str:
        return f"MultiValueLiteral({self.name!r}, {self.value!r})"

    # ------------------------------------------------------------------
    # メンバーシップ計算
    # ------------------------------------------------------------------

    def membership(self, df: pd.DataFrame) -> pd.Series:
        """Return membership scores for this literal."""
        if self.name not in df.columns:
            raise KeyError(
                f"MultiValueLiteral '{self.name}' が DataFrame に存在しません。"
                f"利用可能なカラム: {list(df.columns)}"
            )

        col = df[self.name]

        # カテゴリ値の存在チェック
        if self.value not in col.values:
            warnings.warn(
                f"MultiValueLiteral '{self.name}={self.value}' の値 "
                f"{self.value!r} が DataFrame に存在しません。"
                f"利用可能な値: {sorted(col.dropna().unique().tolist(), key=str)}\n"
                f"型の不一致（例: int vs str）がないか確認してください。",
                UserWarning,
                stacklevel=2,
            )

        return (col == self.value).astype(float)

    # ------------------------------------------------------------------
    # 集合演算ユーティリティ
    # ------------------------------------------------------------------

    def conflicts_with(self, other: object) -> bool:
        """Conflicts with."""
        if not isinstance(other, MultiValueLiteral):
            return False
        return self.name == other.name and self.value != other.value


# ---------------------------------------------------------------------------
# ファクトリ関数
# ---------------------------------------------------------------------------


def parse_literal(text: str) -> SetLiteral | MultiValueLiteral:
    """Parse a literal label into a literal object."""
    text = text.strip()
    if not text:
        raise ValueError("リテラル文字列が空です。")

    if "=" in text:
        # MultiValueLiteral: "name=value"
        name, val_str = text.split("=", 1)
        value: Any = _parse_value(val_str)
        return MultiValueLiteral(name.strip(), value)

    # SetLiteral: "name" または "~name"
    if text.startswith("~"):
        return SetLiteral(text[1:].strip(), negated=True)
    return SetLiteral(text)


def _parse_value(val_str: str) -> Any:
    """Parse a string token into a Python value."""
    val_str = val_str.strip()
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def parse_conjunction(text: str) -> list[SetLiteral | MultiValueLiteral]:
    """Parse a conjunction label into literal objects."""
    if not text.strip():
        raise ValueError("conjunction 文字列が空です。")
    return [parse_literal(part) for part in text.split("*")]
