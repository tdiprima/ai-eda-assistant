"""TEST-1: sessions are keyed by id, never by name."""

import pandas as pd

from conftest import HR_CSV, SALES_CSV


def test_sessions_remain_independent(app, fake_openai):
    first = app.create_session("Data")
    app.upload("sales.csv", SALES_CSV)
    fake_openai.queue.append("1. Which region earns most?\n2. Is revenue skewed?")
    app.generate_questions(objective="revenue drivers")
    first_history = list(app.sessions[first]["question_runs"])
    assert len(first_history) == 1

    second = app.create_session("Data")  # identical name, different dataset
    assert second != first
    app.upload("hr.csv", HR_CSV)

    # Both sessions carry the same display name; selection must still use ids.
    assert [s["name"] for s in app.sessions.values()] == ["Data", "Data"]

    for _ in range(3):
        app.switch_to(first)
        assert app.active_id == first
        assert app.active["filename"] == "sales.csv"
        assert list(app.active["df"].columns) == ["region", "revenue"]
        assert app.active["question_runs"] == first_history

        app.switch_to(second)
        assert app.active_id == second
        assert app.active["filename"] == "hr.csv"
        assert list(app.active["df"].columns) == ["employee", "age", "dept"]
        assert app.active["question_runs"] == []

    # Delete the second (active) session: the first must survive untouched.
    app.switch_to(second)
    app.delete_active()
    assert set(app.sessions) == {first}
    assert app.active_id == first
    survivor = app.sessions[first]
    assert survivor["filename"] == "sales.csv"
    pd.testing.assert_frame_equal(
        survivor["df"], pd.DataFrame({"region": ["East", "West"], "revenue": [100, 250]})
    )
    assert survivor["question_runs"] == first_history
    assert survivor["column_explanations"] == {}
    assert survivor["follow_ups"] == []
    assert len(fake_openai.calls) == 1  # no hidden extra LLM calls


def test_deleting_last_session_returns_to_empty_state(app):
    app.create_session("Only")
    app.delete_active()
    assert app.sessions == {}
    assert app.active_id is None
    assert app.at.title[0].value == "AI EDA Assistant"
