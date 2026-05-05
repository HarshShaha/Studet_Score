"""
Test suite — Student Score MLOps pipeline and API
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PARAMS = {
    "data": {
        "raw_path":      "data/raw/student_performance.csv",
        "processed_dir": "data/processed",
        "test_size":     0.2,
        "random_state":  42,
        "features":      ["Hours Studied", "Sleep Hours", "Attendance"],
        "target":        "Performance Index",
    },
    "preprocessing": {"scale_features": True},
    "models": {
        "ridge":          {"alpha": 1.0},
        "random_forest":  {"n_estimators": 10, "max_depth": 4, "min_samples_split": 5, "random_state": 42},
        "xgboost":        {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                           "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
    },
    "mlflow": {
        "experiment_name": "test_experiment",
        "model_name":      "TestStudentScoreModel",
        "tracking_uri":    "mlruns_test",
    },
}


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe (100 rows)."""
    rng = np.random.default_rng(0)
    n = 100
    study  = rng.uniform(1, 9, n)
    sleep  = rng.uniform(4, 9, n)
    attend = rng.uniform(60, 100, n)
    score  = (10 * study + 0.3 * attend + rng.normal(0, 3, n)).clip(10, 100)
    return pd.DataFrame({
        "Hours Studied":    study.round(2),
        "Sleep Hours":      sleep.round(2),
        "Attendance":       attend.round(2),
        "Performance Index": score.round(2),
    })


# ── Ingest tests ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_synthetic_generation(self, tmp_path):
        from src.pipeline.ingest import generate_synthetic_data
        out = tmp_path / "test.csv"
        generate_synthetic_data(out, n=200)
        df = pd.read_csv(out)
        assert len(df) == 200
        assert "Performance Index" in df.columns
        assert df["Performance Index"].between(10, 100).all()

    def test_validate_data_passes(self, sample_df, tmp_path):
        from src.pipeline.ingest import validate_data
        csv = tmp_path / "data.csv"
        sample_df.to_csv(csv, index=False)
        df = validate_data(csv, ["Hours Studied", "Sleep Hours", "Attendance", "Performance Index"])
        assert len(df) == 100

    def test_validate_data_missing_col(self, sample_df, tmp_path):
        from src.pipeline.ingest import validate_data
        csv = tmp_path / "data.csv"
        sample_df.drop(columns=["Hours Studied"]).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            validate_data(csv, ["Hours Studied", "Performance Index"])


# ── Preprocess tests ──────────────────────────────────────────────────────────

class TestPreprocess:
    def test_split_shapes(self, sample_df, tmp_path, monkeypatch):
        params = {**SAMPLE_PARAMS}
        params["data"] = {**params["data"], "processed_dir": str(tmp_path)}

        from src.pipeline import preprocess as pp_module
        monkeypatch.setattr(pp_module, "load_params", lambda: params)

        from src.pipeline.preprocess import preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess(sample_df, params)

        assert len(X_train) + len(X_test) == 100
        assert len(X_train) == len(y_train)
        assert len(X_test)  == len(y_test)

    def test_scaler_saved(self, sample_df, tmp_path):
        params = {**SAMPLE_PARAMS, "data": {**SAMPLE_PARAMS["data"], "processed_dir": str(tmp_path)}}
        from src.pipeline.preprocess import preprocess
        preprocess(sample_df, params)
        assert (tmp_path / "scaler.joblib").exists()

    def test_feature_columns(self, sample_df, tmp_path):
        params = {**SAMPLE_PARAMS, "data": {**SAMPLE_PARAMS["data"], "processed_dir": str(tmp_path)}}
        from src.pipeline.preprocess import preprocess
        X_train, *_ = preprocess(sample_df, params)
        assert list(X_train.columns) == SAMPLE_PARAMS["data"]["features"]


# ── Training tests ────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_all_models(self, sample_df, tmp_path, monkeypatch):
        params = {
            **SAMPLE_PARAMS,
            "data": {**SAMPLE_PARAMS["data"], "processed_dir": str(tmp_path)},
            "mlflow": {**SAMPLE_PARAMS["mlflow"], "tracking_uri": str(tmp_path / "mlruns")},
        }
        from src.pipeline.preprocess import preprocess
        X_train, X_test, y_train, y_test, _ = preprocess(sample_df, params)

        from src.pipeline.train import train
        results = train(X_train, X_test, y_train, y_test, params)

        assert set(results.keys()) == {"LinearRegression", "Ridge", "RandomForest", "XGBoost"}

    def test_best_model_saved(self, sample_df, tmp_path, monkeypatch):
        params = {
            **SAMPLE_PARAMS,
            "data": {**SAMPLE_PARAMS["data"], "processed_dir": str(tmp_path)},
            "mlflow": {**SAMPLE_PARAMS["mlflow"], "tracking_uri": str(tmp_path / "mlruns")},
        }
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        from src.pipeline.preprocess import preprocess
        X_train, X_test, y_train, y_test, _ = preprocess(sample_df, params)
        from src.pipeline.train import train
        train(X_train, X_test, y_train, y_test, params)
        assert (tmp_path / "models" / "best_model.joblib").exists()


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    """
    These tests use a mock model so they run without requiring a trained artifact.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from sklearn.linear_model import Ridge
        import joblib
        import yaml

        # Write minimal params.yaml
        params = {
            "data": {
                "features": ["Hours Studied", "Sleep Hours", "Attendance"],
                "target": "Performance Index",
                "processed_dir": str(tmp_path),
            },
            "mlflow": {"experiment_name": "x", "model_name": "x", "tracking_uri": "x"},
        }
        params_path = tmp_path / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)

        # Fit & save a quick mock model
        model = Ridge().fit(
            np.array([[7, 8, 90], [3, 6, 70], [5, 7, 80]]),
            np.array([85, 60, 75])
        )
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model, models_dir / "best_model.joblib")

        monkeypatch.chdir(tmp_path)

        from src.api.main import app, load_artifacts
        import asyncio
        asyncio.get_event_loop().run_until_complete(load_artifacts())

        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_predict_valid(self, client):
        r = client.post("/predict", json={"study_hours": 7, "sleep_hours": 8, "attendance": 90})
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["predicted_score"] <= 100
        assert body["grade"] in ["A+", "A", "B", "C", "D", "F"]

    def test_predict_invalid_attendance(self, client):
        r = client.post("/predict", json={"study_hours": 5, "sleep_hours": 7, "attendance": 110})
        assert r.status_code == 422

    def test_predict_boundary_zeros(self, client):
        r = client.post("/predict", json={"study_hours": 0, "sleep_hours": 4, "attendance": 0})
        assert r.status_code == 200
