# AI EDA Assistant

An AI-powered CSV analyzing application that generates insights and exploratory data analysis questions to help you think like a data analyst.

Code from [*"I Built an AI That Thinks Like a Data Analyst — Then It Went Viral. So I Made It Smarter."*](https://medium.com/data-science-collective/i-built-an-ai-that-thinks-like-a-data-analyst-then-it-went-viral-so-i-made-it-smarter-1f3206a8254b) by Mukundan Sankar.

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

1. Upload a CSV file
2. Optionally specify your analysis objective
3. Optionally select key columns to focus on
4. Review the AI-generated questions
5. Download questions as Markdown for documentation

Perfect for data analysts, students, job seekers preparing for take-home projects, and anyone looking to better understand their datasets.

<br>
