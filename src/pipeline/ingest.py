"""
Stage 1 — Data Ingestion
Downloads the Students Performance dataset from Kaggle.
Falls back to synthetic generation if Kaggle credentials are absent.
"""

# import os
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def download_from_kaggle(output_path: Path) -> bool:
    """Attempt Kaggle download; returns True on success."""
    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading dataset from Kaggle...")
        api.dataset_download_files(
            "nikhil7280/student-performance-multiple-linear-regression",
            path=str(output_path.parent),
            unzip=True,
        )
        # Rename if needed
        downloaded = output_path.parent / "Student_Performance.csv"
        if downloaded.exists():
            downloaded.rename(output_path)
        logger.info(f"Dataset saved to {output_path}")
        return True
    except Exception as exc:
        logger.warning(f"Kaggle download failed: {exc}")
        return False


def generate_synthetic_data(output_path: Path, n: int = 10_000, seed: int = 42) -> None:
    """
    Generate synthetic student performance data that mirrors the Kaggle dataset.

    Columns: Hours Studied, Previous Scores, Extracurricular Activities,
             Sleep Hours, Sample Question Papers Practiced, Performance Index
    """
    rng = np.random.default_rng(seed)

    hours_studied    = rng.uniform(1, 9, n)           # 1–9 hrs/day
    sleep_hours      = rng.uniform(4, 9, n)           # 4–9 hrs/night
    attendance       = rng.uniform(60, 100, n)        # 60–100 %
    prev_scores      = rng.uniform(40, 99, n)
    sample_papers    = rng.integers(0, 9, n).astype(float)
    extracurricular  = rng.choice([0, 1], n)          # 0=No, 1=Yes

    # Score formula: study & attendance drive score, sleep has diminishing returns
    noise = rng.normal(0, 3, n)
    performance = (
        10 * hours_studied
        + 0.3 * attendance
        + 0.2 * prev_scores
        + 1.5 * sample_papers
        - 0.5 * extracurricular
        + 2 * np.clip(sleep_hours - 6, -3, 2)        # sleep sweet-spot
        + noise
    )
    performance = np.clip(performance, 10, 100).round(2)

    df = pd.DataFrame({
        "Hours Studied":                     hours_studied.round(2),
        "Previous Scores":                   prev_scores.round(2),
        "Extracurricular Activities":        ["Yes" if x else "No" for x in extracurricular],
        "Sleep Hours":                       sleep_hours.round(2),
        "Sample Question Papers Practiced":  sample_papers.astype(int),
        "Attendance":                        attendance.round(2),
        "Performance Index":                 performance,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset ({n} rows) saved to {output_path}")


def validate_data(path: Path, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    if df.isnull().any().any():
        before = len(df)
        df = df.dropna()
        logger.warning(f"Dropped {before - len(df)} rows with nulls")
    logger.info(f"Data validated: {len(df)} rows, {len(df.columns)} columns")
    return df


def ingest(params: dict | None = None) -> pd.DataFrame:
    if params is None:
        params = load_params()

    raw_path   = Path(params["data"]["raw_path"])
    features   = params["data"]["features"]
    target     = params["data"]["target"]
    required   = features + [target]

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        success = download_from_kaggle(raw_path)
        if not success:
            logger.info("Using synthetic data (set KAGGLE_USERNAME + KAGGLE_KEY for real data)")
            generate_synthetic_data(raw_path, seed=params["data"]["random_state"])

    df = validate_data(raw_path, required)

    # Basic EDA summary
    logger.info("\n--- Data Summary ---")
    logger.info(f"Shape:  {df.shape}")
    logger.info(f"Target stats:\n{df[target].describe().round(2)}")
    return df


if __name__ == "__main__":
    ingest()
