"""Shared fixtures: a fake OpenAI client and helpers to drive the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import llm

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class FakeCompletions:
    """Stands in for client.chat.completions. Records every call.

    `queue` holds canned replies, consumed in order. A str is returned as the
    message content; an Exception is raised; None yields a null content.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.queue: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.queue:
            raise AssertionError("FakeCompletions.create called with no queued reply")
        reply = self.queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    # Convenience -----------------------------------------------------------
    @property
    def last_prompt(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]

    @property
    def last_system(self) -> str:
        return self.calls[-1]["messages"][0]["content"]


@pytest.fixture
def fake_openai(monkeypatch) -> FakeCompletions:
    """Replace the OpenAI class (the only external boundary) with a recorder."""
    completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    return completions


# ---------------------------------------------------------------------------
# App driving helpers
# ---------------------------------------------------------------------------

def _by_label(widgets, label: str):
    for w in widgets:
        if w.label == label:
            return w
    raise LookupError(f"no widget labelled {label!r}; have {[w.label for w in widgets]}")


class App:
    """Thin wrapper over AppTest with named actions for this app."""

    def __init__(self, at: AppTest):
        self.at = at

    # state ---------------------------------------------------------------
    @property
    def sessions(self) -> dict:
        return self.at.session_state["sessions"]

    @property
    def active_id(self):
        return self.at.session_state["active_session_id"]

    @property
    def active(self) -> dict:
        return self.sessions[self.active_id]

    def run(self) -> "App":
        self.at.run()
        assert not self.at.exception, self.at.exception[0].value
        return self

    # sidebar -------------------------------------------------------------
    def create_session(self, name: str) -> int:
        _by_label(self.at.sidebar.text_input, "New session name").set_value(name)
        _by_label(self.at.sidebar.button, "➕ Create session").click()
        self.run()
        return self.active_id

    def switch_to(self, session_id: int) -> "App":
        _by_label(self.at.sidebar.radio, "Active session").set_value(session_id)
        return self.run()

    def delete_active(self) -> "App":
        _by_label(self.at.sidebar.button, "🗑️ Delete active session").click()
        return self.run()

    # main panel ----------------------------------------------------------
    def upload(self, filename: str, content: bytes, session_id=None) -> "App":
        sid = self.active_id if session_id is None else session_id
        self.at.file_uploader(key=f"uploader_{sid}").set_value((filename, content, "text/csv"))
        return self.run()

    def generate_questions(self, objective: str = "", focus: list[str] | None = None) -> "App":
        if objective:
            _by_label(self.at.text_input, "Analysis objective (optional)").set_value(objective)
        if focus:
            _by_label(self.at.multiselect, "Focus columns (optional)").set_value(focus)
        _by_label(self.at.button, "✨ Generate questions").click()
        return self.run()

    def explain_column(self, column: str) -> "App":
        _by_label(self.at.selectbox, "Choose a column").set_value(column)
        _by_label(self.at.button, "💡 Explain column").click()
        return self.run()

    def follow_ups(self, question: str) -> "App":
        _by_label(self.at.selectbox, "Choose a question to dig into").set_value(question)
        _by_label(self.at.button, "↪️ Generate follow-ups").click()
        return self.run()

    # observations --------------------------------------------------------
    @property
    def errors(self) -> list[str]:
        return [e.value for e in self.at.error]


@pytest.fixture
def app(monkeypatch, fake_openai) -> App:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    return App(at).run()


SALES_CSV = b"region,revenue\nEast,100\nWest,250\n"
HR_CSV = b"employee,age,dept\nAna,34,Eng\nBo,45,Ops\nCy,29,Eng\n"
