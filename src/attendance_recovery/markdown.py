"""Tiny Markdown table writer (avoids an extra dependency on ``tabulate``)."""

from __future__ import annotations

import math

import pandas as pd


def _cell(value, digits: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, digits: int = 2, index: bool = False) -> str:
    """Render ``frame`` as a GitHub-flavoured Markdown table."""
    if index:
        frame = frame.reset_index()
    headers = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(v, digits) for v in row) + " |")
    return "\n".join(lines)
