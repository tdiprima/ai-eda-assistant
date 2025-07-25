# streamlit run app.py
import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
from datetime import datetime

MODEL = "gpt-4o"

# Page config
st.set_page_config(page_title="AI Data Question Generator", layout="wide")
st.title("Ask the Right Questions: AI-Powered EDA Assistant")

# Initialize session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'follow_up_questions' not in st.session_state:
    st.session_state.follow_up_questions = {}

# API key handling
api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input(
    "Enter your OpenAI API Key", type="password"
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# Session management sidebar
with st.sidebar:
    st.header("Project Sessions")
    
    # New session
    new_session_name = st.text_input("New Session Name")
    if st.button("Create Session") and new_session_name:
        st.session_state.sessions[new_session_name] = {
            'created': datetime.now().isoformat(),
            'datasets': {},
            'questions': [],
            'follow_ups': {}
        }
        st.session_state.current_session = new_session_name
        st.rerun()
    
    # Select existing session
    if st.session_state.sessions:
        session_options = list(st.session_state.sessions.keys())
        selected_session = st.selectbox("Select Session", session_options, 
                                      index=session_options.index(st.session_state.current_session) if st.session_state.current_session in session_options else 0)
        if selected_session != st.session_state.current_session:
            st.session_state.current_session = selected_session
            st.rerun()
        
        # Delete session
        if st.button("Delete Current Session") and st.session_state.current_session:
            del st.session_state.sessions[st.session_state.current_session]
            st.session_state.current_session = None
            st.rerun()

# Display session history in sidebar
if st.session_state.current_session and st.session_state.sessions[st.session_state.current_session].get('questions'):
    with st.sidebar:
        st.subheader("Session History")
        session_data = st.session_state.sessions[st.session_state.current_session]
        for i, entry in enumerate(session_data['questions']):
            with st.expander(f"Analysis {i+1} - {entry['dataset']}"):
                st.write(f"**Time:** {entry['timestamp'][:19]}")
                st.write(f"**Objective:** {entry.get('objective', 'None')}")
                if entry.get('focus_cols'):
                    st.write(f"**Focus:** {', '.join(entry['focus_cols'])}")

# File upload
if st.session_state.current_session:
    uploaded_file = st.file_uploader(f"Upload CSV for '{st.session_state.current_session}'", type=["csv"])
else:
    st.warning("Please create or select a session first.")
    uploaded_file = None

if uploaded_file and st.session_state.current_session:
    df = pd.read_csv(uploaded_file)
    
    # Store dataset in session
    dataset_name = uploaded_file.name
    st.session_state.sessions[st.session_state.current_session]['datasets'][dataset_name] = {
        'filename': dataset_name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    st.success(f"File '{dataset_name}' uploaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Column explanation feature
    st.subheader("Column Explanation Assistant")
    col_to_explain = st.selectbox("Select a column to explain:", df.columns.tolist())
    if st.button("Explain This Column"):
        with st.spinner("Analyzing column..."):
            col_data = df[col_to_explain]
            col_summary = f"Column '{col_to_explain}' has {col_data.count()} non-null values out of {len(col_data)} total. "
            
            if pd.api.types.is_numeric_dtype(col_data):
                desc = col_data.describe()
                col_summary += f"Numeric column with mean={desc['mean']:.2f}, median={desc['50%']:.2f}, std={desc['std']:.2f}, range=[{desc['min']}, {desc['max']}]. "
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_summary += f"Date column ranging from {col_data.min()} to {col_data.max()}. "
            else:
                unique_vals = col_data.nunique()
                most_common = col_data.value_counts().head(3)
                col_summary += f"Categorical column with {unique_vals} unique values. Most common: {dict(most_common)}. "
            
            explain_prompt = f"""Explain what this column likely represents in a dataset and its potential significance for analysis:
            
Column name: {col_to_explain}
Column summary: {col_summary}
Sample values: {col_data.dropna().head(5).tolist()}
            
Provide insights about what this column might mean, how it could be used in analysis, and any data quality considerations."""
            
            try:
                explanation_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You're a data analyst explaining dataset columns to help users understand their data better."},
                        {"role": "user", "content": explain_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                st.info(explanation_response.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"Error explaining column: {e}")

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

        # GPT Call
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You're a helpful and analytical data assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            questions = response.choices[0].message.content.strip()
            st.markdown(questions)
            
            # Store questions in session
            st.session_state.sessions[st.session_state.current_session]['questions'].append({
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_name,
                'questions': questions,
                'objective': objective,
                'focus_cols': focus_cols
            })
            
            # Follow-up questions section
            st.subheader("Generate Follow-up Questions")
            question_lines = [line.strip() for line in questions.split('\n') if line.strip() and not line.strip().startswith('#')]
            if question_lines:
                selected_question = st.selectbox("Select a question to generate follow-ups for:", question_lines)
                if st.button("Generate Follow-up Questions"):
                    with st.spinner("Generating follow-up questions..."):
                        followup_prompt = f"""Based on this data analysis question: "{selected_question}"
                        
And this dataset context: {summary_text}
                        
Generate 5 specific follow-up questions that would help dive deeper into this analysis. Focus on actionable insights and practical next steps."""
                        
                        try:
                            followup_response = client.chat.completions.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You're an expert data analyst who helps generate insightful follow-up questions for deeper analysis."},
                                    {"role": "user", "content": followup_prompt}
                                ],
                                temperature=0.6,
                                max_tokens=400
                            )
                            follow_ups = followup_response.choices[0].message.content.strip()
                            st.markdown("**Follow-up Questions:**")
                            st.markdown(follow_ups)
                            
                            # Store follow-ups
                            if selected_question not in st.session_state.sessions[st.session_state.current_session]['follow_ups']:
                                st.session_state.sessions[st.session_state.current_session]['follow_ups'][selected_question] = []
                            st.session_state.sessions[st.session_state.current_session]['follow_ups'][selected_question].append({
                                'timestamp': datetime.now().isoformat(),
                                'follow_ups': follow_ups
                            })
                        except Exception as e:
                            st.error(f"Error generating follow-ups: {e}")
            
            # Auto-generate summary report
            st.subheader("Summary Report")
            if st.button("Generate Summary Report"):
                with st.spinner("Creating summary report..."):
                    session_data = st.session_state.sessions[st.session_state.current_session]
                    
                    report_prompt = f"""Create a comprehensive markdown summary report for this data analysis session:
                    
Dataset: {dataset_name}
Shape: {df.shape[0]} rows, {df.shape[1]} columns
Objective: {objective if objective else 'General exploratory analysis'}
Focus columns: {', '.join(focus_cols) if focus_cols else 'All columns'}
                    
Data summary:
{summary_text}
                    
Generated questions:
{questions}
                    
Create a professional report that includes:
1. Executive Summary
2. Dataset Overview
3. Key Questions for Analysis
4. Recommended Next Steps
5. Data Quality Observations
                    
Format it as a clean, professional markdown document."""
                    
                    try:
                        report_response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You're a senior data analyst creating professional analysis reports."},
                                {"role": "user", "content": report_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1200
                        )
                        report_content = report_response.choices[0].message.content.strip()
                        st.markdown("**Generated Summary Report:**")
                        st.markdown(report_content)
                        
                        # Download report
                        report_filename = f"{st.session_state.current_session}_{dataset_name.replace('.csv', '')}_report.md"
                        st.download_button(
                            "Download Summary Report",
                            data=report_content,
                            file_name=report_filename,
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
            
            # Enhanced Markdown Export
            session_data = st.session_state.sessions[st.session_state.current_session]
            md_export = f"""# AI-Generated EDA Questions - {st.session_state.current_session}

**Dataset:** {dataset_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Objective:** {objective if objective else 'General exploratory analysis'}

## Initial Questions
{questions}

## Follow-up Questions
"""
            
            for question, followups in session_data.get('follow_ups', {}).items():
                if followups:
                    md_export += f"\n### {question}\n{followups[-1]['follow_ups']}\n"
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Questions (Markdown)", 
                    data=f"# AI-Generated EDA Questions\n\n{questions}", 
                    file_name="eda_questions.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    "Download Complete Analysis", 
                    data=md_export, 
                    file_name=f"{st.session_state.current_session}_complete_analysis.md",
                    mime="text/markdown"
                )

        except Exception as e:
            st.error(f"Error: {e}")
