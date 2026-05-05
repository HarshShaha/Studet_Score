"""
Stage 2 — Preprocessing
Selects features, scales them, splits into train/test, and persists artifacts.
"""

import logging
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame | None = None, params: dict | None = None):
    """
    Returns (X_train, X_test, y_train, y_test, scaler).
    Persists split CSVs and the fitted scaler to disk.
    """
    if params is None:
        params = load_params()

    if df is None:
        df = pd.read_csv(params["data"]["raw_path"])

    features   = params["data"]["features"]
    target     = params["data"]["target"]
    test_size  = params["data"]["test_size"]
    seed       = params["data"]["random_state"]
    scale      = params["preprocessing"]["scale_features"]
    out_dir    = Path(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df[features].copy()
    y = df[target].copy()

    logger.info(f"Features: {features}")
    logger.info(f"Target  : {target}  |  range [{y.min():.1f}, {y.max():.1f}]")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    logger.info(f"Split — train: {len(X_train)}  test: {len(X_test)}")

    # Scaling
    scaler = StandardScaler()
    if scale:
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=features, index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=features, index=X_test.index
        )
        logger.info("Features scaled with StandardScaler")
    else:
        X_train_s, X_test_s = X_train, X_test

    # Persist splits
    X_train_s.assign(**{target: y_train}).to_csv(out_dir / "train.csv", index=False)
    X_test_s.assign(**{target: y_test}).to_csv(out_dir / "test.csv", index=False)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    logger.info(f"Artifacts saved to {out_dir}/")
    return X_train_s, X_test_s, y_train, y_test, scaler


if __name__ == "__main__":
    from src.pipeline.ingest import ingest
    df = ingest()
    preprocess(df)
