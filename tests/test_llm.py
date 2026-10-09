"""TEST-3: AI workflows pass the right context, store results, and fail loudly."""

import openai

try:  # openai>=3 ships its own httpx fork
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
import pytest

from conftest import HR_CSV
from data_profile import describe_column, describe_dataset


@pytest.fixture
def loaded(app):
    app.create_session("People")
    app.upload("hr.csv", HR_CSV)
    return app


def _timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


def _auth_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("Incorrect API key provided", response=response, body=None)


# -- successful paths --------------------------------------------------------

def test_generate_questions_passes_context_and_stores_result(loaded, fake_openai):
    reply = "1. How does age vary by dept?\n2. Is Eng older than Ops?"
    fake_openai.queue.append(reply)
    loaded.generate_questions(objective="retention risk", focus=["age", "dept"])

    assert len(fake_openai.calls) == 1
    call = fake_openai.calls[0]
    assert call["model"] == "gpt-4o-mini"
    prompt = fake_openai.last_prompt
    assert describe_dataset(loaded.active["df"]) in prompt
    assert "Analysis objective: retention risk" in prompt
    assert "Focus columns: age, dept" in prompt
    assert "Write 10 exploratory data analysis questions" in prompt

    runs = loaded.active["question_runs"]
    assert len(runs) == 1
    assert runs[0]["objective"] == "retention risk"
    assert runs[0]["focus_columns"] == ["age", "dept"]
    assert runs[0]["questions"] == reply
    assert runs[0]["timestamp"]
    assert loaded.at.markdown[0].value == reply


def test_generate_questions_defaults_when_no_objective_or_focus(loaded, fake_openai):
    fake_openai.queue.append("1. Q")
    loaded.generate_questions()
    prompt = fake_openai.last_prompt
    assert "Analysis objective: general exploration" in prompt
    assert "Focus columns: none specified" in prompt


def test_explain_column_passes_context_and_stores_result(loaded, fake_openai):
    fake_openai.queue.append("Age in years of each employee.")
    loaded.explain_column("age")

    df = loaded.active["df"]
    prompt = fake_openai.last_prompt
    assert "Column to explain: age" in prompt
    assert f"Column details: {describe_column(df, 'age')}" in prompt
    assert describe_dataset(df) in prompt
    assert loaded.active["column_explanations"] == {"age": "Age in years of each employee."}
    assert loaded.at.info[0].value == "Age in years of each employee."


def test_follow_ups_pass_context_and_store_result(loaded, fake_openai):
    fake_openai.queue += ["1. Does dept predict age?\n2. Any outliers in age?", "1. Which dept is youngest?"]
    loaded.generate_questions()
    loaded.follow_ups("Any outliers in age?")

    prompt = fake_openai.last_prompt
    assert "Starting question: Any outliers in age?" in prompt
    assert describe_dataset(loaded.active["df"]) in prompt
    assert "Write 5 follow-up questions" in prompt
    assert loaded.active["follow_ups"] == [
        {"question": "Any outliers in age?", "follow_ups": "1. Which dept is youngest?"}
    ]


def test_client_is_built_with_the_entered_key(loaded, fake_openai, monkeypatch):
    import llm

    client = llm.get_client("sk-abc")
    assert client.api_key == "sk-abc"


# -- failure paths -----------------------------------------------------------

FAILURES = [
    pytest.param(_timeout, id="timeout"),
    pytest.param(_auth_error, id="auth"),
    pytest.param(lambda: "", id="empty-string"),
    pytest.param(lambda: "   \n", id="whitespace"),
    pytest.param(lambda: None, id="null-content"),
]


def _assert_actionable_error(app):
    assert not app.at.exception, "unhandled exception reached the user"
    assert app.errors, "expected a visible error message"
    msg = app.errors[0].lower()
    assert any(word in msg for word in ("try again", "check", "api key", "retry")), msg


@pytest.mark.parametrize("make", FAILURES)
def test_question_generation_failure_is_reported_not_stored(loaded, fake_openai, make):
    fake_openai.queue.append(make())
    loaded.generate_questions(objective="x")
    _assert_actionable_error(loaded)
    assert loaded.active["question_runs"] == []


@pytest.mark.parametrize("make", FAILURES)
def test_column_explanation_failure_is_reported_not_stored(loaded, fake_openai, make):
    fake_openai.queue.append(make())
    loaded.explain_column("dept")
    _assert_actionable_error(loaded)
    assert loaded.active["column_explanations"] == {}


@pytest.mark.parametrize("make", FAILURES)
def test_follow_up_failure_is_reported_not_stored(loaded, fake_openai, make):
    fake_openai.queue += ["1. Seed question?", make()]
    loaded.generate_questions()
    loaded.follow_ups("Seed question?")
    _assert_actionable_error(loaded)
    assert loaded.active["follow_ups"] == []
    assert len(loaded.active["question_runs"]) == 1  # earlier success untouched
