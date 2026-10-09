"""
Build Markdown text for downloads.

Two kinds of export:
1. Just the latest list of questions.
2. A full summary report of everything done in a session.
"""

from datetime import datetime


def questions_markdown(session: dict, run: dict) -> str:
    """Markdown for a single question-generation run."""
    lines = [
        f"# EDA Questions: {session['name']}",
        "",
        f"- **File:** {session['filename']}",
        f"- **Generated:** {run['timestamp']}",
        f"- **Objective:** {run['objective'] or '(none)'}",
        f"- **Focus columns:** {', '.join(run['focus_columns']) or '(none)'}",
        "",
        run["questions"],
        "",
    ]
    return "\n".join(lines)


def session_report_markdown(session: dict, dataset_summary: str) -> str:
    """Markdown report covering the whole session."""
    lines = [
        f"# EDA Session Report: {session['name']}",
        "",
        f"- **File:** {session['filename']}",
        f"- **Report created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Dataset overview",
        "",
        "```",
        dataset_summary,
        "```",
        "",
    ]

    # Every question-generation run, oldest first.
    lines.append("## Generated questions")
    lines.append("")
    if not session["question_runs"]:
        lines.append("_No questions generated yet._")
    for i, run in enumerate(session["question_runs"], start=1):
        lines.append(f"### Run {i} ({run['timestamp']})")
        lines.append(f"- **Objective:** {run['objective'] or '(none)'}")
        lines.append(f"- **Focus columns:** {', '.join(run['focus_columns']) or '(none)'}")
        lines.append("")
        lines.append(run["questions"])
        lines.append("")

    # Column explanations.
    lines.append("## Column explanations")
    lines.append("")
    if not session["column_explanations"]:
        lines.append("_No columns explained yet._")
    for column, text in session["column_explanations"].items():
        lines.append(f"### {column}")
        lines.append(text)
        lines.append("")

    # Follow-up questions.
    lines.append("## Follow-up questions")
    lines.append("")
    if not session["follow_ups"]:
        lines.append("_No follow-ups generated yet._")
    for item in session["follow_ups"]:
        lines.append(f"### Starting question")
        lines.append(item["question"])
        lines.append("")
        lines.append("**Follow-ups:**")
        lines.append(item["follow_ups"])
        lines.append("")

    return "\n".join(lines)
