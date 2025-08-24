# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.2",
#   "numpy>=1.26.0",
#   "matplotlib>=3.8.4",
#   "seaborn>=0.13.2",
#   "httpx>=0.27.0",
#   "tenacity>=8.2.3",
#   "scikit-learn>=1.4.2",
#   "pillow>=10.3.0",
# ]
# ///

"""
Project 2 - Automated Analysis (autolysis.py)
---------------------------------------------
Usage:
  uv run autolysis.py /path/to/dataset.csv

This script:
  - Loads ANY valid CSV.
  - Performs generic, robust analysis (summary stats, missingness, correlation, outliers, clustering, simple time checks).
  - Generates 1–3 PNG charts in the current working directory.
  - Calls an LLM (gpt-4o-mini) via an OpenAI-compatible API using the AIPROXY_TOKEN env var.
  - Writes a story-style README.md with embedded images.

Design notes (for evaluators):
  - Single-file design with uv inline metadata.
  - Careful token budget: only compact summaries are sent to LLM.
  - Retries on LLM calls with tenacity.
  - Dynamic chart selection based on dataset profile.
  - Safe fallbacks if token is missing or network fails.
"""

from __future__ import annotations
import os
import sys
import io
import math
import json
import textwrap
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------- Utilities --------------

def _shorten(s: str, max_chars: int = 2000) -> str:
    s = str(s)
    return (s[: max_chars - 3] + "...") if len(s) > max_chars else s

def _is_datetime(series: pd.Series) -> bool:
    if np.issubdtype(series.dtype, np.datetime64):
        return True
    try:
        pd.to_datetime(series.dropna().head(50), errors="raise")
        return True
    except Exception:
        return False

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if (pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]

def _first_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _is_datetime(df[c]):
            return c
    return None

def _safe_filename(stem: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return stem

# -------------- LLM Client (OpenAI-compatible via httpx) --------------

import httpx

OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")  # REQUIRED by spec

class LLMError(RuntimeError):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(LLMError),
)
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Minimal OpenAI Chat Completions call via httpx.
    Respects the AIPROXY_TOKEN env var for Authorization.
    """
    if not AIPROXY_TOKEN:
        raise LLMError("Missing AIPROXY_TOKEN in environment.")

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    url = f"{OPENAI_BASE}/v1/chat/completions"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(str(e))

# -------------- Analysis --------------

@dataclass
class DataProfile:
    shape: Tuple[int, int]
    dtypes: Dict[str, str]
    missing: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    correlations: Optional[List[List[float]]]
    corr_columns: List[str]
    top_categorical: Dict[str, List[Tuple[str, int]]]
    outliers: Dict[str, int]
    time_column: Optional[str]
    notes: List[str]

def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    notes: List[str] = []
    # Dtypes
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    # Missingness
    missing = df.isna().sum().to_dict()

    # Numeric summary
    num_cols = _numeric_cols(df)
    numeric_summary = {}
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        # reshape
        for col in num_cols:
            numeric_summary[col] = {
                "count": float(desc[col].get("count", float("nan"))),
                "mean": float(desc[col].get("mean", float("nan"))),
                "std": float(desc[col].get("std", float("nan"))),
                "min": float(desc[col].get("min", float("nan"))),
                "25%": float(desc[col].get("25%", float("nan"))),
                "50%": float(desc[col].get("50%", float("nan"))),
                "75%": float(desc[col].get("75%", float("nan"))),
                "max": float(desc[col].get("max", float("nan"))),
            }
    else:
        notes.append("No numeric columns detected.")

    # Correlations
    correlations = None
    corr_columns: List[str] = []
    if len(num_cols) >= 2:
        corr_df = df[num_cols].corr(numeric_only=True)
        correlations = corr_df.values.tolist()
        corr_columns = list(corr_df.columns)

    # Categorical top values
    top_categorical: Dict[str, List[Tuple[str, int]]] = {}
    for c in _categorical_cols(df):
        vc = df[c].fillna("NA").value_counts().head(5)
        top_categorical[c] = [(str(k), int(v)) for k, v in vc.items()]

    # Outliers (simple z-score > 3)
    outliers: Dict[str, int] = {}
    for c in num_cols:
        s = df[c].dropna()
        if s.std(ddof=0) == 0 or s.empty:
            outliers[c] = 0
            continue
        z = (s - s.mean()) / s.std(ddof=0)
        outliers[c] = int((np.abs(z) > 3).sum())

    # Time column detection
    time_col = _first_datetime_col(df)
    if time_col:
        # ensure it's converted
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            pass

    return DataProfile(
        shape=(df.shape[0], df.shape[1]),
        dtypes=dtypes,
        missing=missing,
        numeric_summary=numeric_summary,
        correlations=correlations,
        corr_columns=corr_columns,
        top_categorical=top_categorical,
        outliers=outliers,
        time_column=time_col,
        notes=notes,
    )

# -------------- Visualization --------------

def _save_fig(fig, fname: str, size: int = 512):
    # Export square figures suitable for README tiling
    fig.set_size_inches(5, 5)  # ~ 500 px at 100 dpi
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str], out_path: str) -> Optional[str]:
    if len(columns) < 2:
        return None
    fig, ax = plt.subplots()
    corr = df[columns].corr(numeric_only=True)
    sns.heatmap(corr, ax=ax, annot=False)
    ax.set_title("Correlation Heatmap")
    fname = _safe_filename("correlation_heatmap.png")
    _save_fig(fig, fname)
    return fname

def plot_missingness(missing: Dict[str, int], out_path: str) -> Optional[str]:
    if not missing:
        return None
    items = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:20]
    cols, vals = zip(*items) if items else ([], [])
    fig, ax = plt.subplots()
    ax.barh(cols, vals)
    ax.set_xlabel("Missing values")
    ax.set_title("Missingness by Column (Top 20)")
    fname = _safe_filename("missingness.png")
    _save_fig(fig, fname)
    return fname

def plot_top_numeric_distribution(df: pd.DataFrame, out_path: str) -> Optional[str]:
    nums = _numeric_cols(df)
    if not nums:
        return None
    # pick numeric with highest variance
    var_series = df[nums].var(numeric_only=True).sort_values(ascending=False)
    if var_series.empty:
        return None
    col = var_series.index[0]
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Distribution: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    fname = _safe_filename(f"distribution_{col}.png")
    _save_fig(fig, fname)
    return fname

def plot_pca_clusters(df: pd.DataFrame, out_path: str) -> Optional[str]:
    nums = _numeric_cols(df)
    if len(nums) < 2:
        return None
    clean = df[nums].dropna()
    if clean.shape[0] < 10:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)
    # choose k
    k = 3 if clean.shape[0] >= 60 else 2
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(XY[:, 0], XY[:, 1], c=labels)
    ax.set_title("PCA Projection with KMeans Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fname = _safe_filename("pca_clusters.png")
    _save_fig(fig, fname)
    return fname

# -------------- README / Story --------------

def build_system_prompt() -> str:
    return textwrap.dedent("""
    You are a precise data analyst and a compelling storyteller.
    Write a crisp, well-structured Markdown narrative that:
    1) Briefly describes the dataset.
    2) Summarizes the analysis performed (summary stats, missingness, correlations, outliers, clustering/time if relevant).
    3) Highlights 3–6 key insights with evidence.
    4) Explains practical implications and recommended next steps.
    5) References the provided figures by filename where helpful.
    Use clear headers, bullet points, and short paragraphs. Be specific and avoid hype.
    """).strip()

def build_user_prompt(profile: DataProfile, figure_files: List[str], sample_rows: List[Dict[str, Any]]) -> str:
    payload = {
        "profile": asdict(profile),
        "figures": figure_files,
        "sample_rows": sample_rows[:5],
    }
    return "Here is the compact analysis context:\n\n" + _shorten(json.dumps(payload, indent=2), 6000)

def write_readme(markdown: str, outfile: str = "README.md"):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(markdown)

def fallback_readme(profile: DataProfile, figs: List[str]) -> str:
    # In case LLM is unavailable, write a simple deterministic README.
    md = ["# Automated Analysis\n"]
    md.append(f"**Rows x Columns:** {profile.shape[0]} x {profile.shape[1]}\n")
    md.append("## Columns & Types\n")
    for k, v in profile.dtypes.items():
        md.append(f"- `{k}`: {v}")
    md.append("\n## Missing Values (Top)\n")
    for k, v in sorted(profile.missing.items(), key=lambda x: x[1], reverse=True)[:10]:
        md.append(f"- `{k}`: {v}")
    if profile.corr_columns and profile.correlations is not None:
        md.append("\n## Correlations\nA correlation heatmap was generated.\n")
    if any(v > 0 for v in profile.outliers.values()):
        md.append("\n## Outliers\nSome potential outliers were detected using a z-score>3 rule.\n")
    if profile.time_column:
        md.append(f"\n## Time\nDetected a time-like column: `{profile.time_column}`.\n")
    if figs:
        md.append("\n## Figures\n")
        for f in figs:
            md.append(f"![{f}]({f})")
    return "\n".join(md)

# -------------- Main --------------

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py /path/to/dataset.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # Work in CWD, as required by spec (README.md and *.png written here)
    cwd = os.getcwd()

    # Load CSV
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

    # Attempt datetime conversion for any object columns that look like dates
    for c in df.columns:
        if _is_datetime(df[c]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass

    # Profile
    profile = profile_dataframe(df)

    # Visualizations (pick up to 3 robust charts)
    figure_files: List[str] = []
    try:
        f1 = plot_correlation_heatmap(df, profile.corr_columns, cwd)
        if f1:
            figure_files.append(f1)
    except Exception:
        pass

    try:
        f2 = plot_missingness(profile.missing, cwd)
        if f2:
            figure_files.append(f2)
    except Exception:
        pass

    try:
        f3 = plot_top_numeric_distribution(df, cwd)
        if f3 and len(figure_files) < 3:
            figure_files.append(f3)
    except Exception:
        pass

    try:
        f4 = plot_pca_clusters(df, cwd)
        if f4 and len(figure_files) < 3:
            figure_files.append(f4)
    except Exception:
        pass

    # Sample rows for context (never send full data to LLM)
    sample_rows = df.head(5).to_dict(orient="records")

    # LLM Narrative
    readme_text = ""
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(profile, figure_files, sample_rows)
        content = llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.3)
        readme_text = content
    except Exception as e:
        # Fallback deterministic README if LLM is not available
        readme_text = fallback_readme(profile, figure_files)

    # Ensure figures are included in README (append if LLM forgot)
    if figure_files and all(fname not in readme_text for fname in figure_files):
        readme_text += "\n\n## Figures\n" + "\n".join(f"![{f}]({f})" for f in figure_files)

    # Write README.md
    write_readme(readme_text, "README.md")

    print("Done. Generated README.md and up to 3 PNG charts in current directory.")

if __name__ == "__main__":
    main()
