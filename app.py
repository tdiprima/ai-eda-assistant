# streamlit run app.py
import streamlit as st
import pandas as pd
from openai import OpenAI

# Page config
st.set_page_config(page_title="AI Data Question Generator", layout="wide")
st.title("Ask the Right Questions: AI-Powered EDA Assistant")

# API key handling
api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input(
    "Enter your OpenAI API Key", type="password"
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Custom objective input
    objective = st.text_input("Optional: What's your goal or business objective with this dataset?",
                              placeholder="E.g. Understand what drives customer churn")

    # Column focus selector
    focus_cols = st.multiselect("Optional: Pick key columns to focus the AI's questions on",
                                df.columns.tolist())

    st.subheader("AI-Generated Questions")

    with st.spinner("Analyzing data and generating questions..."):
        summary_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                summary_lines.append(
                    f"- {col} (numeric): mean={desc['mean']:.2f}, min={desc['min']}, max={desc['max']}, std={desc['std']:.2f}"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary_lines.append(
                    f"- {col} (date): range={df[col].min()} to {df[col].max()}"
                )
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                unique_vals = df[col].nunique()
                sample_vals = df[col].dropna().unique()[:3]
                summary_lines.append(
                    f"- {col} (categorical): {unique_vals} unique values, e.g. {sample_vals}"
                )
            else:
                summary_lines.append(f"- {col} ({dtype})")

        summary_text = "\n".join(summary_lines)

        # Build the prompt
        prompt = f"""
You're a data analyst reviewing a new dataset. Here's a summary of the columns:
{summary_text}
"""

        if objective:
            prompt += f"\nThe user's stated objective is: {objective}"

        if focus_cols:
            prompt += f"\nFocus especially on these columns: {', '.join(focus_cols)}"

        prompt += """
Based on this structure, suggest 10 insightful questions a data analyst or business user should explore to better understand this dataset. Think about trends, segments, outliers, and relationships.
"""

        # GPT-4 Call
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful and analytical data assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            questions = response.choices[0].message.content.strip()
            st.markdown(questions)

            # Markdown Export
            md_export = f"# AI-Generated EDA Questions\n\n{questions}"
            st.download_button(
                "Download Questions (Markdown)", 
                data=md_export, 
                file_name="eda_questions.md"
            )

        except Exception as e:
            st.error(f"Error: {e}")