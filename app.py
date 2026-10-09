"""
AI EDA Assistant - a Streamlit app.

Upload a CSV, and let OpenAI suggest exploratory data analysis questions.
Work is organised into "sessions" (one per dataset) in the sidebar.

Run with:  streamlit run app.py
"""

import os
import re
from datetime import datetime

import openai
import pandas as pd
import streamlit as st

import llm
import report
from data_profile import describe_column, describe_dataset

st.set_page_config(page_title="AI EDA Assistant", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Session storage
#
# All sessions live in st.session_state["sessions"], a dict that maps
# a session id -> a session dict. A session dict looks like:
#
#   {
#       "name": "Sales data",
#       "filename": "sales.csv",
#       "file_id": "<streamlit upload id>",
#       "df": <DataFrame>,
#       "question_runs": [ {timestamp, objective, focus_columns, questions}, ... ],
#       "column_explanations": { "price": "This column ...", ... },
#       "follow_ups": [ {question, follow_ups}, ... ],
#   }
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Create the keys we need in session_state if they are missing."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    if "next_session_id" not in st.session_state:
        st.session_state.next_session_id = 1


def create_session(name: str) -> None:
    """Add a new, empty session and make it active."""
    session_id = st.session_state.next_session_id
    st.session_state.next_session_id += 1

    st.session_state.sessions[session_id] = {
        "name": name,
        "filename": None,
        "file_id": None,
        "df": None,
        "question_runs": [],
        "column_explanations": {},
        "follow_ups": [],
    }
    st.session_state.active_session_id = session_id
    st.session_state.session_picker = session_id


def delete_session(session_id: int) -> None:
    """Remove a session. If it was active, switch to another one."""
    st.session_state.sessions.pop(session_id, None)
    if st.session_state.active_session_id == session_id:
        remaining = list(st.session_state.sessions)
        st.session_state.active_session_id = remaining[0] if remaining else None
        st.session_state.session_picker = st.session_state.active_session_id


def get_active_session() -> dict | None:
    session_id = st.session_state.active_session_id
    return st.session_state.sessions.get(session_id)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_numbered_list(markdown_text: str) -> list[str]:
    """
    Turn a numbered Markdown list into a list of plain strings.
    "1. What is X?" -> "What is X?"
    """
    questions = []
    for line in markdown_text.splitlines():
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions


def call_llm(fn, *args) -> str | None:
    """Run one LLM call. Show an actionable error and return None on failure."""
    try:
        text = fn(*args)
    except openai.APITimeoutError:
        st.error("OpenAI did not answer in time. Please try again.")
        return None
    except openai.AuthenticationError:
        st.error("OpenAI rejected the API key. Check the key in the sidebar and try again.")
        return None
    except openai.OpenAIError as exc:
        st.error(f"OpenAI request failed: {exc}. Please try again.")
        return None
    if not text:
        st.error("OpenAI returned an empty response. Please try again.")
        return None
    return text


def get_api_key() -> str:
    """Read the OpenAI key from the environment, or let the user type it."""
    key = os.environ.get("OPENAI_API_KEY", "")
    key = st.sidebar.text_input("OpenAI API key", value=key, type="password")
    return key


# ---------------------------------------------------------------------------
# Sidebar: API key + session management
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.title("🔍 AI EDA Assistant")
    api_key = get_api_key()

    st.sidebar.divider()
    st.sidebar.subheader("Sessions")

    # Create a new session.
    new_name = st.sidebar.text_input("New session name", placeholder="e.g. Q3 sales")
    if st.sidebar.button("➕ Create session", use_container_width=True):
        create_session(new_name.strip() or f"Session {st.session_state.next_session_id}")
        st.rerun()

    # Pick which session is active.
    sessions = st.session_state.sessions
    if sessions:
        ids = list(sessions)
        # The radio is driven through its own session_state key. Passing a
        # changing `index` instead would give the widget a new identity on
        # every switch and silently drop the user's next click.
        if st.session_state.get("session_picker") not in ids:
            st.session_state.session_picker = st.session_state.active_session_id
        st.session_state.active_session_id = st.sidebar.radio(
            "Active session",
            ids,
            key="session_picker",
            # Streamlit matches radio choices by label, so two sessions with
            # the same name would collapse into one. Append the id to keep
            # every label unique.
            format_func=lambda i: f"{sessions[i]['name']} (#{i})",
        )

        # Delete the active session. Done in an on_click callback so the
        # radio's session_state key can be reset before the radio is drawn.
        st.sidebar.button(
            "🗑️ Delete active session",
            use_container_width=True,
            on_click=delete_session,
            args=(st.session_state.active_session_id,),
        )

        # Short history of what has happened in this session.
        session = get_active_session()
        st.sidebar.divider()
        st.sidebar.subheader("History")
        if session["filename"]:
            st.sidebar.write(f"📄 {session['filename']}")
        for run in reversed(session["question_runs"]):
            st.sidebar.caption(
                f"🕒 {run['timestamp']}: questions"
                + (f" about *{run['objective']}*" if run["objective"] else "")
            )
        for column in session["column_explanations"]:
            st.sidebar.caption(f"💡 explained `{column}`")
        for item in session["follow_ups"]:
            st.sidebar.caption(f"↪️ follow-ups for: {item['question'][:40]}...")
    else:
        st.sidebar.info("Create a session to get started.")

    return api_key


# ---------------------------------------------------------------------------
# Main panel pieces
# ---------------------------------------------------------------------------

def render_upload(session: dict, session_id: int) -> None:
    """Upload a CSV and store it in the session."""
    # Keyed by session so switching sessions never carries a file across.
    uploaded = st.file_uploader(
        "Upload a CSV file", type=["csv"], key=f"uploader_{session_id}"
    )
    # Compare by upload identity, not filename, so a revised file with the
    # same name is still picked up.
    if uploaded is not None and uploaded.file_id != session["file_id"]:
        try:
            df = pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read {uploaded.name}: {exc}. Upload a valid, non-empty CSV.")
            return
        # A new dataset invalidates everything derived from the old one.
        session["df"] = df
        session["filename"] = uploaded.name
        session["file_id"] = uploaded.file_id
        session["question_runs"] = []
        session["column_explanations"] = {}
        session["follow_ups"] = []
        st.success(f"Loaded {uploaded.name}")

    if session["df"] is not None:
        df = session["df"]
        st.caption(f"{session['filename']}: {df.shape[0]} rows x {df.shape[1]} columns")
        with st.expander("Preview data"):
            st.dataframe(df.head(50), use_container_width=True)


def render_question_generator(session: dict, client, dataset_summary: str) -> None:
    """Objective + focus columns -> EDA questions."""
    st.subheader("1. Generate EDA questions")
    df = session["df"]

    objective = st.text_input(
        "Analysis objective (optional)",
        placeholder="e.g. Understand what drives customer churn",
    )
    focus_columns = st.multiselect("Focus columns (optional)", options=list(df.columns))
    how_many = st.slider("Number of questions", 5, 20, 10)

    if st.button("✨ Generate questions", type="primary"):
        with st.spinner("Asking OpenAI..."):
            questions = call_llm(
                llm.generate_questions, client, dataset_summary, objective, focus_columns, how_many
            )
        if questions:
            session["question_runs"].append(
                {
                    "timestamp": now_text(),
                    "objective": objective,
                    "focus_columns": focus_columns,
                    "questions": questions,
                }
            )

    # Show the most recent run and offer a Markdown download.
    if session["question_runs"]:
        latest = session["question_runs"][-1]
        st.markdown(latest["questions"])
        st.download_button(
            "⬇️ Export questions as Markdown",
            data=report.questions_markdown(session, latest),
            file_name="eda_questions.md",
            mime="text/markdown",
        )

        # Older runs are tucked away so the page stays clean.
        if len(session["question_runs"]) > 1:
            with st.expander("Previous runs"):
                for run in reversed(session["question_runs"][:-1]):
                    st.markdown(f"**{run['timestamp']}** - objective: {run['objective'] or '(none)'}")
                    st.markdown(run["questions"])
                    st.divider()


def render_column_explainer(session: dict, client, dataset_summary: str) -> None:
    """Pick a column and get a plain-English explanation of it."""
    st.subheader("2. Column explainer")
    df = session["df"]

    column = st.selectbox("Choose a column", options=list(df.columns))
    if st.button("💡 Explain column"):
        with st.spinner("Asking OpenAI..."):
            explanation = call_llm(
                llm.explain_column, client, column, describe_column(df, column), dataset_summary
            )
        if explanation:
            session["column_explanations"][column] = explanation

    # Show the explanation for the currently selected column, if we have one.
    if column in session["column_explanations"]:
        st.info(session["column_explanations"][column])


def render_follow_ups(session: dict, client, dataset_summary: str) -> None:
    """Pick one generated question and get deeper follow-up questions."""
    st.subheader("3. Follow-up questions")

    if not session["question_runs"]:
        st.caption("Generate some questions first.")
        return

    # Collect every question from every run, newest first, without duplicates.
    all_questions: list[str] = []
    for run in reversed(session["question_runs"]):
        for q in split_numbered_list(run["questions"]):
            if q not in all_questions:
                all_questions.append(q)

    chosen = st.selectbox("Choose a question to dig into", options=all_questions)
    if st.button("↪️ Generate follow-ups"):
        with st.spinner("Asking OpenAI..."):
            follow_ups = call_llm(llm.generate_follow_ups, client, chosen, dataset_summary)
        if follow_ups:
            session["follow_ups"].append({"question": chosen, "follow_ups": follow_ups})

    # Show follow-ups for the chosen question, if any exist.
    for item in reversed(session["follow_ups"]):
        if item["question"] == chosen:
            st.markdown(item["follow_ups"])
            break


def render_report_download(session: dict, dataset_summary: str) -> None:
    """Download everything from the session as one Markdown report."""
    st.subheader("4. Session report")
    st.download_button(
        "⬇️ Download Markdown summary report",
        data=report.session_report_markdown(session, dataset_summary),
        file_name=f"eda_report_{session['name'].replace(' ', '_')}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    api_key = render_sidebar()

    session = get_active_session()
    if session is None:
        st.title("AI EDA Assistant")
        st.write("👈 Create a session in the sidebar to begin.")
        return

    st.title(session["name"])
    render_upload(session, st.session_state.active_session_id)

    if session["df"] is None:
        st.info("Upload a CSV to continue.")
        return

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to use the AI features.")
        return

    client = llm.get_client(api_key)
    dataset_summary = describe_dataset(session["df"])

    st.divider()
    render_question_generator(session, client, dataset_summary)
    st.divider()

    left, right = st.columns(2)
    with left:
        render_column_explainer(session, client, dataset_summary)
    with right:
        render_follow_ups(session, client, dataset_summary)

    st.divider()
    render_report_download(session, dataset_summary)


if __name__ == "__main__":
    main()
