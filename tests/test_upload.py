"""TEST-2: replacing a dataset invalidates derived history; bad uploads keep old data."""

import pandas as pd
import pytest

from conftest import SALES_CSV

REVISED_CSV = b"region,revenue,units\nNorth,999,7\n"


def _seed_history(app, fake_openai):
    fake_openai.queue += ["1. Which region earns most?", "Revenue is money.", "1. Why East?"]
    app.generate_questions(objective="growth", focus=["revenue"])
    app.explain_column("revenue")
    app.follow_ups("Which region earns most?")
    assert len(app.active["question_runs"]) == 1
    assert app.active["column_explanations"] == {"revenue": "Revenue is money."}
    assert len(app.active["follow_ups"]) == 1


def test_replacement_invalidates_history(app, fake_openai):
    app.create_session("S")
    app.upload("sales.csv", SALES_CSV)
    _seed_history(app, fake_openai)

    # Plain rerun (same upload): history must survive.
    app.run()
    assert len(app.active["question_runs"]) == 1
    assert app.active["column_explanations"] == {"revenue": "Revenue is money."}
    assert len(app.active["follow_ups"]) == 1
    assert list(app.active["df"].columns) == ["region", "revenue"]

    # Same filename, new contents: data updates, history clears.
    app.upload("sales.csv", REVISED_CSV)
    pd.testing.assert_frame_equal(
        app.active["df"], pd.DataFrame({"region": ["North"], "revenue": [999], "units": [7]})
    )
    assert app.active["filename"] == "sales.csv"
    assert app.active["question_runs"] == []
    assert app.active["column_explanations"] == {}
    assert app.active["follow_ups"] == []
    assert not app.errors


@pytest.mark.parametrize(
    "label, content",
    [
        ("empty", b""),
        ("malformed", b'a,b\n1,2,3\n"unterminated\n'),
        ("invalid-encoding", b"col\n\xff\xfe\xfa\n"),
    ],
)
def test_bad_upload_shows_error_and_keeps_previous_data(app, fake_openai, label, content):
    app.create_session("S")
    app.upload("sales.csv", SALES_CSV)
    _seed_history(app, fake_openai)
    before = app.active["df"].copy()

    app.upload("sales.csv", content)

    assert app.errors, f"{label}: expected an error message"
    assert "sales.csv" in app.errors[0]
    assert "valid, non-empty CSV" in app.errors[0]
    pd.testing.assert_frame_equal(app.active["df"], before)
    assert app.active["filename"] == "sales.csv"
    assert len(app.active["question_runs"]) == 1
    assert app.active["column_explanations"] == {"revenue": "Revenue is money."}
    assert len(app.active["follow_ups"]) == 1
