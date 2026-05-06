"""
Stage 3 — Model Training + MLflow Tracking
Trains multiple regressors, logs each to MLflow, registers the best model.
"""

import logging
import yaml
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def build_models(params: dict) -> dict:
    mp = params["models"]
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=mp["ridge"]["alpha"]),
        "RandomForest": RandomForestRegressor(
            n_estimators=mp["random_forest"]["n_estimators"],
            max_depth=mp["random_forest"]["max_depth"],
            min_samples_split=mp["random_forest"]["min_samples_split"],
            random_state=mp["random_forest"]["random_state"],
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=mp["xgboost"]["n_estimators"],
            max_depth=mp["xgboost"]["max_depth"],
            learning_rate=mp["xgboost"]["learning_rate"],
            subsample=mp["xgboost"]["subsample"],
            colsample_bytree=mp["xgboost"]["colsample_bytree"],
            random_state=mp["xgboost"]["random_state"],
            verbosity=0,
        ),
    }


def train_and_log(
    model_name: str,
    model,
    X_train, y_train,
    X_test,  y_test,
    params:   dict,
    features: list[str],
) -> tuple[float, str]:
    """Fit model, log to MLflow, return (r2_score, run_id)."""
    # mlp = params["mlflow"]

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)

        # Log params & metrics
        mlflow.log_params({"model": model_name, "features": str(features)})
        mlflow.log_metrics(metrics)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))

        # Log model artifact
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            f"  {model_name:<20} RMSE={metrics['rmse']:.3f}  "
            f"MAE={metrics['mae']:.3f}  R²={metrics['r2']:.4f}"
        )
        return metrics["r2"], run.info.run_id


def register_best_model(run_id: str, model_name: str, params: dict) -> None:
    """Register the best run in MLflow Model Registry."""
    mlp = params["mlflow"]
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=mlp["model_name"])
    logger.info(
        f"Registered '{mlp['model_name']}' v{mv.version} "
        f"(run={run_id[:8]}…) from {model_name}"
    )


def train(
    X_train=None, X_test=None, y_train=None, y_test=None,
    params: dict | None = None,
) -> dict:
    if params is None:
        params = load_params()

    # Load from disk if not passed directly
    if X_train is None:
        out_dir  = Path(params["data"]["processed_dir"])
        features = params["data"]["features"]
        target   = params["data"]["target"]
        train_df = pd.read_csv(out_dir / "train.csv")
        test_df  = pd.read_csv(out_dir / "test.csv")
        X_train, y_train = train_df[features], train_df[target]
        X_test,  y_test  = test_df[features],  test_df[target]

    features = params["data"]["features"]
    mlp      = params["mlflow"]

    # MLflow setup
    mlflow.set_tracking_uri(mlp["tracking_uri"])
    mlflow.set_experiment(mlp["experiment_name"])

    models    = build_models(params)
    results   = {}
    best_r2   = -999.0
    best_run  = None
    best_name = None

    logger.info(f"\n{'='*55}")
    logger.info("Training models — tracking with MLflow")
    logger.info(f"{'='*55}")

    for name, model in models.items():
        r2, run_id = train_and_log(
            name, model, X_train, y_train, X_test, y_test, params, features
        )
        results[name] = {"r2": r2, "run_id": run_id}
        if r2 > best_r2:
            best_r2, best_run, best_name = r2, run_id, name

    logger.info(f"\nBest model: {best_name}  R²={best_r2:.4f}")
    register_best_model(best_run, best_name, params)

    # Save best model locally
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_model = models[best_name]
    joblib.dump(best_model, models_dir / "best_model.joblib")
    logger.info("Best model saved to models/best_model.joblib")

    return results


if __name__ == "__main__":
    train()
