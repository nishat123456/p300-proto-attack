"""Speller layouts and the flash-group representation.

A flash group is the set of symbols lit by one flash. Row/column spellers are
the special case where each group is one row or one column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    layout_id: str
    symbols: str
    n_rows: int
    n_cols: int

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def index(self, symbol: str) -> int:
        i = self.symbols.find(symbol)
        if i < 0:
            raise KeyError(f"symbol {symbol!r} not in layout {self.layout_id}")
        return i

    def row_col(self, symbol: str) -> tuple[int, int]:
        return divmod(self.index(symbol), self.n_cols)


# BCI Competition III Dataset II: codes 1-6 = columns (left to right), 7-12 = rows (top to bottom).
COMP3 = Layout("rc6x6_comp3", "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_", 6, 6)


def rc_code_to_group(layout: Layout, code: int) -> np.ndarray:
    """Boolean mask over symbols lit by a row/column stimulus code (Comp III convention)."""
    group = np.zeros(layout.n_symbols, dtype=bool)
    if 1 <= code <= layout.n_cols:
        group[code - 1 :: layout.n_cols] = True
    elif layout.n_cols < code <= layout.n_cols + layout.n_rows:
        r = code - layout.n_cols - 1
        group[r * layout.n_cols : (r + 1) * layout.n_cols] = True
    else:
        raise ValueError(f"invalid row/column code {code}")
    return group


def symbol_rc_codes(layout: Layout, symbol: str) -> tuple[int, int]:
    """(row_code, col_code) for a symbol under the Comp III convention."""
    r, c = layout.row_col(symbol)
    return layout.n_cols + 1 + r, c + 1
