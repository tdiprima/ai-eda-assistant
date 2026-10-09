"""
Turn a pandas DataFrame into a short, plain-text description.

The LLM never sees the whole CSV. It sees this summary instead,
which keeps prompts small and avoids sending every row to OpenAI.
"""

import pandas as pd


def describe_column(df: pd.DataFrame, column: str) -> str:
    """Return a one-line summary of a single column."""
    series = df[column]
    dtype = str(series.dtype)
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))

    # Show a few example values so the model knows what the data looks like.
    examples = series.dropna().astype(str).unique()[:5]
    examples_text = ", ".join(examples)

    line = f"- {column} (type={dtype}, unique={unique}, missing={missing})"
    line += f" examples: {examples_text}"

    # Numeric columns get min/mean/max too.
    if pd.api.types.is_numeric_dtype(series) and series.notna().any():
        line += (
            f" | min={series.min():.4g}, "
            f"mean={series.mean():.4g}, "
            f"max={series.max():.4g}"
        )
    return line


def describe_dataset(df: pd.DataFrame, max_columns: int = 60) -> str:
    """Return a multi-line summary of the whole dataset."""
    rows, cols = df.shape
    lines = [f"Rows: {rows}", f"Columns: {cols}", "", "Column details:"]

    for column in df.columns[:max_columns]:
        lines.append(describe_column(df, column))

    if cols > max_columns:
        lines.append(f"... and {cols - max_columns} more columns not shown.")

    return "\n".join(lines)
