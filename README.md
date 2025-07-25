# AI EDA Assistant

An AI-powered CSV analyzing application that generates insights and exploratory data analysis questions to help you think like a data analyst.

Some code adapted from the article [*"I Built an AI That Thinks Like a Data Analyst — Then It Went Viral. So I Made It Smarter."*](https://medium.com/data-science-collective/i-built-an-ai-that-thinks-like-a-data-analyst-then-it-went-viral-so-i-made-it-smarter-1f3206a8254b) by Mukundan Sankar.

## Features

- Upload CSV files for analysis
- AI-generated exploratory questions tailored to your dataset
- Custom objective input to focus questions on specific goals
- Column focus selector to zoom in on key variables
- Export questions to Markdown format
- Clean, intuitive Streamlit interface

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   streamlit run app.py
   ```

3. Set your OpenAI API key in Streamlit secrets: `.streamlit/secrets.toml`

## Usage

1. Create or select a project session
2. Upload a CSV file
3. Get AI-generated analysis questions
4. Click on any column to get an explanation
5. Generate follow-up questions for deeper analysis
6. Create and download summary reports

Perfect for data analysts, students, job seekers preparing for take-home projects, and anyone looking to better understand their datasets.

## CSV

Credit to Janos Hajagos' Synthea Data: [visit_occurrence.csv](https://raw.githubusercontent.com/jhajagos/SyntheaData520/refs/heads/main/visit_occurrence.csv)

<br>
