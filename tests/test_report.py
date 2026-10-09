"""TEST-5: Markdown exports are complete, attributed, and ordered."""

import re

from report import questions_markdown, session_report_markdown


def _session(**overrides) -> dict:
    base = {
        "name": "Q3 sales",
        "filename": "sales.csv",
        "file_id": "f1",
        "df": None,
        "question_runs": [],
        "column_explanations": {},
        "follow_ups": [],
    }
    base.update(overrides)
    return base


SUMMARY = "Rows: 2\nColumns: 2\n\nColumn details:\n- region (...)"


def test_empty_history_report():
    text = session_report_markdown(_session(), SUMMARY)
    lines = text.splitlines()
    assert lines[0] == "# EDA Session Report: Q3 sales"
    assert "- **File:** sales.csv" in lines
    assert any(re.fullmatch(r"- \*\*Report created:\*\* \d{4}-\d{2}-\d{2} \d{2}:\d{2}", l) for l in lines)
    assert "## Dataset overview" in lines
    assert "```\n" + SUMMARY + "\n```" in text
    assert "_No questions generated yet._" in lines
    assert "_No columns explained yet._" in lines
    assert "_No follow-ups generated yet._" in lines
    assert "### Run" not in text


def test_populated_report_is_complete_and_ordered():
    session = _session(
        question_runs=[
            {
                "timestamp": "2026-01-01 09:00:00",
                "objective": "",
                "focus_columns": [],
                "questions": "1. First run Q1\n2. First run Q2",
            },
            {
                "timestamp": "2026-01-02 10:30:00",
                "objective": "churn",
                "focus_columns": ["region", "revenue"],
                "questions": "1. Second run Q1",
            },
        ],
        column_explanations={"region": "Sales territory.", "revenue": "Money earned."},
        follow_ups=[
            {"question": "First run Q1", "follow_ups": "1. FU-A\n2. FU-B"},
            {"question": "Second run Q1", "follow_ups": "1. FU-C"},
        ],
    )
    text = session_report_markdown(session, SUMMARY)

    expected_fragments = [
        "# EDA Session Report: Q3 sales",
        "- **File:** sales.csv",
        "## Dataset overview",
        SUMMARY,
        "## Generated questions",
        "### Run 1 (2026-01-01 09:00:00)",
        "- **Objective:** (none)",
        "- **Focus columns:** (none)",
        "1. First run Q1\n2. First run Q2",
        "### Run 2 (2026-01-02 10:30:00)",
        "- **Objective:** churn",
        "- **Focus columns:** region, revenue",
        "1. Second run Q1",
        "## Column explanations",
        "### region\nSales territory.",
        "### revenue\nMoney earned.",
        "## Follow-up questions",
        "### Starting question\nFirst run Q1\n\n**Follow-ups:**\n1. FU-A\n2. FU-B",
        "### Starting question\nSecond run Q1\n\n**Follow-ups:**\n1. FU-C",
    ]
    positions = [text.index(fragment) for fragment in expected_fragments]
    assert positions == sorted(positions), "report sections/runs are out of order"

    for placeholder in ("_No questions generated yet._", "_No columns explained yet._", "_No follow-ups generated yet._"):
        assert placeholder not in text
    assert text.count("### Run ") == 2
    assert text.count("### Starting question") == 2


def test_report_attributes_to_its_own_session_only():
    a = _session(name="Alpha", filename="a.csv", column_explanations={"x": "from A"})
    b = _session(name="Beta", filename="b.csv", column_explanations={"y": "from B"})
    report_a = session_report_markdown(a, "summary A")
    report_b = session_report_markdown(b, "summary B")
    assert "Alpha" in report_a and "a.csv" in report_a and "from A" in report_a and "summary A" in report_a
    for foreign in ("Beta", "b.csv", "from B", "summary B"):
        assert foreign not in report_a
    assert "Alpha" not in report_b and "a.csv" not in report_b


def test_questions_markdown_single_run():
    run = {
        "timestamp": "2026-03-04 05:06:07",
        "objective": "pricing",
        "focus_columns": ["price", "qty"],
        "questions": "1. Does price drive qty?",
    }
    expected = (
        "# EDA Questions: Q3 sales\n"
        "\n"
        "- **File:** sales.csv\n"
        "- **Generated:** 2026-03-04 05:06:07\n"
        "- **Objective:** pricing\n"
        "- **Focus columns:** price, qty\n"
        "\n"
        "1. Does price drive qty?\n"
    )
    assert questions_markdown(_session(), run) == expected


def test_questions_markdown_defaults_for_blank_fields():
    run = {"timestamp": "t", "objective": "", "focus_columns": [], "questions": "1. Q"}
    text = questions_markdown(_session(), run)
    assert "- **Objective:** (none)" in text
    assert "- **Focus columns:** (none)" in text
