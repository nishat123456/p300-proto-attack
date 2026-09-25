import numpy as np

from p3a.data.layout import COMP3, rc_code_to_group, symbol_rc_codes


def test_grid():
    assert COMP3.n_symbols == 36
    assert COMP3.row_col("A") == (0, 0)
    assert COMP3.row_col("_") == (5, 5)
    assert symbol_rc_codes(COMP3, "O") == (9, 3)  # row 3 -> code 9, col 3 -> code 3


def test_every_symbol_in_exactly_one_row_and_one_col():
    groups = np.stack([rc_code_to_group(COMP3, c) for c in range(1, 13)])
    assert (groups.sum(axis=1) == 6).all()
    assert (groups.sum(axis=0) == 2).all()


def test_group_matches_codes():
    for s in COMP3.symbols:
        r, c = symbol_rc_codes(COMP3, s)
        k = COMP3.index(s)
        assert rc_code_to_group(COMP3, r)[k] and rc_code_to_group(COMP3, c)[k]
