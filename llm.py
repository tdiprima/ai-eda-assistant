"""
All calls to OpenAI live here.

Each function builds a prompt, sends it, and returns plain text.
Keeping the prompts in one place makes them easy to read and tweak.
"""

from openai import OpenAI

MODEL = "gpt-4o-mini"


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def ask(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Send one request to OpenAI and return the reply text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


# ---------------------------------------------------------------------------
# Feature 1: generate EDA questions
# ---------------------------------------------------------------------------

def generate_questions(
    client: OpenAI,
    dataset_summary: str,
    objective: str,
    focus_columns: list[str],
    how_many: int = 10,
) -> str:
    system_prompt = (
        "You are a senior data analyst. You write sharp, specific "
        "exploratory data analysis (EDA) questions for a dataset."
    )

    focus_text = ", ".join(focus_columns) if focus_columns else "none specified"
    objective_text = objective.strip() or "general exploration"

    user_prompt = f"""
Here is a summary of the dataset:

{dataset_summary}

Analysis objective: {objective_text}
Focus columns: {focus_text}

Write {how_many} exploratory data analysis questions tailored to this dataset.
Rules:
- Refer to real column names from the summary.
- If focus columns are given, most questions should involve them.
- Mix question types: distributions, relationships, outliers, missing data, time trends (if dates exist), and group comparisons.
- Return a numbered Markdown list. One question per line. No extra commentary.
""".strip()

    return ask(client, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Feature 2: explain what a column likely represents
# ---------------------------------------------------------------------------

def explain_column(client: OpenAI, column_name: str, column_summary: str, dataset_summary: str) -> str:
    system_prompt = (
        "You are a data analyst who explains dataset columns in plain English."
    )

    user_prompt = f"""
Dataset summary (for context):

{dataset_summary}

Column to explain: {column_name}
Column details: {column_summary}

In 3 to 5 short sentences, explain:
1. What this column most likely represents.
2. How it might be used in analysis.
3. Any data quality concerns you notice (missing values, odd ranges, etc.).
Be honest if you are guessing.
""".strip()

    return ask(client, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Feature 3: follow-up questions for one chosen question
# ---------------------------------------------------------------------------

def generate_follow_ups(client: OpenAI, question: str, dataset_summary: str, how_many: int = 5) -> str:
    system_prompt = (
        "You are a senior data analyst who helps people dig deeper into a finding."
    )

    user_prompt = f"""
Dataset summary:

{dataset_summary}

Starting question: {question}

Write {how_many} follow-up questions an analyst would naturally ask next.
Each should go one step deeper or check an assumption behind the starting question.
Return a numbered Markdown list. One question per line. No extra commentary.
""".strip()

    return ask(client, system_prompt, user_prompt)
