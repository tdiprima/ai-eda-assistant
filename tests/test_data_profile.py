"""TEST-4: profiling statistics, edge cases, and size limits."""

import numpy as np
import pandas as pd
import pytest

from data_profile import MAX_EXAMPLE_CHARS, MAX_SUMMARY_CHARS, describe_column, describe_dataset


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param(
            [1, 2, 3, 4],
            "- x (type=int64, unique=4, missing=0) examples: 1, 2, 3, 4 | min=1, mean=2.5, max=4",
            id="ints",
        ),
        pytest.param(
            [1.0, None, 3.0],
            "- x (type=float64, unique=2, missing=1) examples: 1.0, 3.0 | min=1, mean=2, max=3",
            id="floats-with-nan",
        ),
        pytest.param(
            ["b", "a", "b", None],
            "- x (type=object, unique=2, missing=1) examples: b, a",
            id="strings",
        ),
        pytest.param(
            [True, False, True],
            "- x (type=bool, unique=2, missing=0) examples: True, False | min=0, mean=0.6667, max=1",
            id="bools",
        ),
        pytest.param(
            [None, None],
            "- x (type=object, unique=0, missing=2) examples: ",
            id="all-null",
        ),
        pytest.param(
            [np.nan, np.nan],
            "- x (type=float64, unique=0, missing=2) examples: ",
            id="all-nan-numeric",
        ),
        pytest.param(
            [],
            "- x (type=float64, unique=0, missing=0) examples: ",
            id="empty",
        ),
        pytest.param(
            ["café", "日本語", "😀"],
            "- x (type=object, unique=3, missing=0) examples: café, 日本語, 😀",
            id="unicode",
        ),
    ],
)
def test_describe_column_known_values(values, expected):
    df = pd.DataFrame({"x": values})
    assert describe_column(df, "x") == expected


def test_describe_column_shows_at_most_five_examples():
    df = pd.DataFrame({"x": list("abcdefgh")})
    assert "examples: a, b, c, d, e" in describe_column(df, "x")
    assert "f" not in describe_column(df, "x").split("examples:")[1]


@pytest.mark.parametrize("length", [39, 40, 41, 200])
def test_example_values_never_exceed_limit(length):
    value = "v" * length
    line = describe_column(pd.DataFrame({"x": [value]}), "x")
    example = line.split("examples: ")[1]
    assert len(example) <= MAX_EXAMPLE_CHARS
    if length <= MAX_EXAMPLE_CHARS:
        assert example == value
    else:
        assert example == "v" * (MAX_EXAMPLE_CHARS - 3) + "..."
        assert len(example) == MAX_EXAMPLE_CHARS


def test_unicode_example_truncation_counts_characters_not_bytes():
    value = "日" * 50
    line = describe_column(pd.DataFrame({"x": [value]}), "x")
    example = line.split("examples: ")[1]
    assert example == "日" * 37 + "..."


def test_describe_dataset_structure():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    text = describe_dataset(df)
    lines = text.splitlines()
    assert lines[:4] == ["Rows: 2", "Columns: 2", "", "Column details:"]
    assert lines[4] == describe_column(df, "a")
    assert lines[5] == describe_column(df, "b")
    assert len(lines) == 6


def test_describe_dataset_empty_frame():
    text = describe_dataset(pd.DataFrame())
    assert text.splitlines() == ["Rows: 0", "Columns: 0", "", "Column details:"]


def _wide(n_cols: int) -> pd.DataFrame:
    return pd.DataFrame({f"c{i}": [i] for i in range(n_cols)})


def test_sixty_columns_all_shown():
    text = describe_dataset(_wide(60))
    assert text.count("\n- c") == 60
    assert "more columns not shown" not in text


def test_sixty_one_columns_truncates_with_notice():
    text = describe_dataset(_wide(61))
    assert text.count("\n- c") == 60
    assert "- c59 " in text
    assert "- c60 " not in text
    assert text.splitlines()[-1] == "... and 1 more columns not shown."


def test_max_columns_parameter():
    text = describe_dataset(_wide(5), max_columns=2)
    assert text.count("\n- c") == 2
    assert text.splitlines()[-1] == "... and 3 more columns not shown."


def test_dataset_summary_capped_at_limit():
    # 60 columns with long names and long string values comfortably exceeds 8,000 chars.
    df = pd.DataFrame({("column_" * 10) + str(i): ["x" * 40] * 2 for i in range(60)})
    text = describe_dataset(df)
    assert len(text) == MAX_SUMMARY_CHARS
    assert text.endswith("...")


def test_dataset_summary_just_under_limit_is_not_truncated():
    df = _wide(3)
    text = describe_dataset(df)
    assert len(text) < MAX_SUMMARY_CHARS
    assert not text.endswith("...")
