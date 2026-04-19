"""
Entry point — run preprocessing then analysis.

Usage::

    python main.py

Preprocessing artifacts are written to ``data/processed/``.
Analysis outputs (plots + tables) are written to ``output/``.

Both :func:`run_preprocessing` and :func:`run_analysis` are parameterised
functions that can be called independently from a backend API (e.g. FastAPI).
"""

from src.pipeline import run_analysis, run_preprocessing

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"

if __name__ == "__main__":
    data = run_preprocessing(RAW_DIR, PROCESSED_DIR)
    run_analysis(OUTPUT_DIR, data=data)
