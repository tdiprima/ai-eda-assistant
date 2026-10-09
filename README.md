# AI EDA Assistant 📊

A Streamlit app that reads your CSV and uses OpenAI to tell you **what questions to ask of your data** before you start plotting.

## The problem it solves

You get a new dataset. You open it, see 40 columns, and freeze. What matters? Where do you start? Experienced analysts carry a mental checklist of exploratory data analysis (EDA) questions. This app gives you that checklist, tailored to *your* columns and *your* goal.

## Example use case

A marketing analyst receives `customers.csv` with signup dates, plan tiers, monthly spend, support tickets, and a churn flag. Their manager asks: "Why are customers leaving?"

1. **Create a session** named "Churn investigation" and upload the CSV.
2. **Set the objective** to "Understand what drives customer churn" and pick `churned`, `plan_tier`, and `support_tickets` as focus columns.
3. **Generate questions.** The app returns a list such as:
   - How does churn rate differ across plan tiers?
   - Is there a threshold of support tickets after which churn jumps?
   - Do customers who signed up in a particular month churn more?
4. **Explain a column.** Unsure what `plan_tier` codes mean? The column explainer looks at its values and tells you it is likely a categorical subscription level, and flags that 12% of rows are missing.
5. **Dig deeper.** Pick "Is there a threshold of support tickets after which churn jumps?" and generate follow-ups, such as whether ticket *type* or *resolution time* matters more than count.
6. **Export.** Download the questions as Markdown for your notebook, or download the full session report to share with your manager.

The analyst now has a focused plan instead of a blank page.

## Features

| Feature | What it does |
|---|---|
| Sessions | Sidebar create/delete. Each session holds one CSV plus its full history. |
| Question generation | Objective and focus columns steer the questions. Adjustable count. |
| Markdown export | One click to download the latest question list. |
| Column explainer | Plain-English guess at what a column represents, plus quality concerns. |
| Follow-up questions | Pick any generated question and get deeper questions. |
| Session report | Downloadable Markdown summary of everything done in the session. |

## Privacy note

The app never sends your raw rows to OpenAI. It sends a compact summary: column names, types, counts, a few example values, and basic statistics. See `data_profile.py`.

## Setup

```bash
uv sync
export OPENAI_API_KEY=sk-...
uv run streamlit run app.py
```

Or without `uv`: `pip install .` then `streamlit run app.py`.

You can also paste the key into the sidebar instead of setting the environment variable.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI and session state |
| `llm.py` | All OpenAI prompts and calls (change the model here) |
| `data_profile.py` | Turns a DataFrame into the text summary sent to the model |
| `report.py` | Builds the Markdown for both download buttons |

The code favours clarity over cleverness. Each file has one job, and each function has a short docstring saying what it does.

<br>
