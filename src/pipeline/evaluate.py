"""
Stage 4 — Evaluation
Loads the best saved model, runs detailed evaluation, prints a comparison report.
"""

import logging
import yaml
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def evaluate(results: dict | None = None, params: dict | None = None) -> dict:
    if params is None:
        params = load_params()

    out_dir   = Path(params["data"]["processed_dir"])
    features  = params["data"]["features"]
    target    = params["data"]["target"]
    model_path = Path("models/best_model.joblib")

    test_df = pd.read_csv(out_dir / "test.csv")
    X_test  = test_df[features]
    y_test  = test_df[target]

    model   = joblib.load(model_path)
    preds   = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    residuals = y_test - preds
    within_5  = np.mean(np.abs(residuals) <= 5) * 100
    within_10 = np.mean(np.abs(residuals) <= 10) * 100

    metrics = {
        "rmse":       round(rmse, 4),
        "mae":        round(mae, 4),
        "r2":         round(r2, 4),
        "within_5pts": round(within_5, 2),
        "within_10pts": round(within_10, 2),
        "n_test":     int(len(y_test)),
    }

    logger.info("\n" + "="*50)
    logger.info("FINAL MODEL EVALUATION REPORT")
    logger.info("="*50)
    logger.info(f"  RMSE              : {rmse:.3f} points")
    logger.info(f"  MAE               : {mae:.3f} points")
    logger.info(f"  R²                : {r2:.4f}")
    logger.info(f"  Within ±5 pts     : {within_5:.1f}%")
    logger.info(f"  Within ±10 pts    : {within_10:.1f}%")
    logger.info(f"  Test samples      : {len(y_test)}")
    logger.info("="*50)

    # Sample predictions table
    sample = pd.DataFrame({
        "study_hours": X_test["Hours Studied"].values[:5].round(1),
        "sleep_hours": X_test["Sleep Hours"].values[:5].round(1),
        "attendance":  X_test["Attendance"].values[:5].round(1),
        "actual":      y_test.values[:5].round(1),
        "predicted":   preds[:5].round(1),
        "error":       (y_test.values[:5] - preds[:5]).round(2),
    })
    logger.info("\nSample predictions:\n" + sample.to_string(index=False))

    # Persist metrics
    metrics_path = Path("models/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
