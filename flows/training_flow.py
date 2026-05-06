"""
Prefect Training Flow
Orchestrates: ingest → preprocess → train → evaluate
Run with:  python flows/training_flow.py
Or deploy: prefect deploy flows/training_flow.py:training_flow
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
# import logging
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact

# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="load-params", retries=1)
def load_params_task() -> dict:
    logger = get_run_logger()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    logger.info("Params loaded")
    return params


@task(name="ingest-data", retries=2, retry_delay_seconds=10)
def ingest_task(params: dict):
    logger = get_run_logger()
    logger.info("Stage 1 — Data ingestion")
    from src.pipeline.ingest import ingest
    df = ingest(params)
    logger.info(f"Ingested {len(df)} rows")
    return df


@task(name="preprocess-data")
def preprocess_task(df, params: dict):
    logger = get_run_logger()
    logger.info("Stage 2 — Preprocessing")
    from src.pipeline.preprocess import preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df, params)
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@task(name="train-models", retries=1)
def train_task(X_train, X_test, y_train, y_test, params: dict) -> dict:
    logger = get_run_logger()
    logger.info("Stage 3 — Model training + MLflow tracking")
    from src.pipeline.train import train
    results = train(X_train, X_test, y_train, y_test, params)
    logger.info(f"Trained {len(results)} models")
    return results


@task(name="evaluate-model")
def evaluate_task(results: dict, params: dict) -> dict:
    logger = get_run_logger()
    logger.info("Stage 4 — Evaluation")
    from src.pipeline.evaluate import evaluate
    metrics = evaluate(results, params)
    return metrics


@task(name="log-artifacts")
def log_artifacts_task(results: dict, metrics: dict):
    logger = get_run_logger()

    # Comparison table artifact
    rows = [
        {"model": name, "r2": f"{info['r2']:.4f}", "run_id": info["run_id"][:8] + "…"}
        for name, info in results.items()
    ]
    create_table_artifact(
        key="model-comparison",
        table={"columns": ["model", "r2", "run_id"], "data": [[r["model"], r["r2"], r["run_id"]] for r in rows]},
        description="Model comparison — all runs",
    )

    # Final metrics markdown artifact
    md = f"""## Final Model Metrics

| Metric | Value |
|--------|-------|
| RMSE | {metrics['rmse']} |
| MAE | {metrics['mae']} |
| R² | {metrics['r2']} |
| Within ±5 pts | {metrics['within_5pts']}% |
| Within ±10 pts | {metrics['within_10pts']}% |
| Test samples | {metrics['n_test']} |
"""
    create_markdown_artifact(key="final-metrics", markdown=md, description="Best model metrics")
    logger.info("Artifacts logged to Prefect UI")


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(
    name="student-score-training-flow",
    description="End-to-end MLOps pipeline for student score prediction",
    version="1.0",
    log_prints=True,
)
def training_flow():
    params = load_params_task()

    df = ingest_task(params)

    X_train, X_test, y_train, y_test = preprocess_task(df, params)

    results = train_task(X_train, X_test, y_train, y_test, params)

    metrics = evaluate_task(results, params)

    log_artifacts_task(results, metrics)

    return metrics


# ── Scheduled flow (example: daily at 2 AM) ───────────────────────────────────

@flow(
    name="student-score-scheduled-flow",
    description="Daily retraining flow",
    version="1.0",
)
def scheduled_training_flow():
    """
    Same pipeline; attach this to a Prefect Deployment for scheduled runs.
    
    Deploy with:
        prefect deploy flows/training_flow.py:scheduled_training_flow \
            --name daily-retrain \
            --cron "0 2 * * *"
    """
    return training_flow()  


if __name__ == "__main__":
    result = training_flow()
    print(f"\nFlow complete — final metrics: {result}")
