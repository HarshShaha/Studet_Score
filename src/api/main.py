"""
FastAPI — Student Score Prediction API
Loads the best trained model and serves predictions at /predict.
"""

import joblib
import yaml
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Score Prediction API",
    description=(
        "Predict a student's exam Performance Index (0–100) "
        "given study hours, sleep hours, and attendance."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    study_hours: float = Field(..., ge=0, le=24,  description="Daily study hours (0–24)")
    sleep_hours: float = Field(..., ge=0, le=24,  description="Nightly sleep hours (0–24)")
    attendance:  float = Field(..., ge=0, le=100, description="Attendance percentage (0–100)")

    @field_validator("study_hours", "sleep_hours")
    @classmethod
    def hours_reasonable(cls, v: float) -> float:
        if v > 16:
            raise ValueError("Hours value seems unrealistic (>16). Please check your input.")
        return v


class PredictResponse(BaseModel):
    predicted_score: float = Field(..., description="Predicted Performance Index (0–100)")
    grade:           str   = Field(..., description="Letter grade")
    confidence_note: str   = Field(..., description="Model reliability note")
    inputs:          dict  = Field(..., description="Echo of input features")


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str


# ── Globals ───────────────────────────────────────────────────────────────────
MODEL  = None
SCALER = None
PARAMS = None

GRADE_THRESHOLDS = [
    (90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D"), (0, "F")
]


def grade(score: float) -> str:
    for threshold, letter in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_artifacts():
    global MODEL, SCALER, PARAMS

    with open("params.yaml") as f:
        PARAMS = yaml.safe_load(f)

    model_path  = Path("models/best_model.joblib")
    scaler_path = Path(PARAMS["data"]["processed_dir"]) / "scaler.joblib"

    if not model_path.exists():
        raise RuntimeError(
            "No trained model found at models/best_model.joblib. "
            "Run the training flow first."
        )

    MODEL  = joblib.load(model_path)
    SCALER = joblib.load(scaler_path) if scaler_path.exists() else None

    model_type = type(MODEL).__name__
    print(f"✓ Model loaded  : {model_type}")
    print(f"✓ Scaler loaded : {SCALER is not None}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Meta"], include_in_schema=False)
def root():
    """Serve the web frontend."""
    frontend = Path("src/frontend/index.html")
    if frontend.exists():
        return FileResponse(str(frontend))
    return {
        "service": "Student Score Prediction",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # features = PARAMS["data"]["features"]   # ['Hours Studied', 'Sleep Hours', 'Attendance']
    X_raw = np.array([[payload.study_hours, payload.sleep_hours, payload.attendance]])

    X = SCALER.transform(X_raw) if SCALER is not None else X_raw
    score_raw = float(MODEL.predict(X)[0])
    score = round(min(max(score_raw, 0.0), 100.0), 2)

    # Confidence note based on input range
    if payload.attendance < 65:
        note = "Low attendance may limit model accuracy."
    elif payload.study_hours < 2:
        note = "Very low study hours — prediction may be at the lower bound."
    else:
        note = "Inputs are within typical training range."

    return PredictResponse(
        predicted_score=score,
        grade=grade(score),
        confidence_note=note,
        inputs={
            "study_hours": payload.study_hours,
            "sleep_hours": payload.sleep_hours,
            "attendance":  payload.attendance,
        },
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(payloads: list[PredictRequest]):
    """Predict scores for multiple students at once."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(payloads) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500")

    X_raw = np.array([
        [p.study_hours, p.sleep_hours, p.attendance] for p in payloads
    ])
    X      = SCALER.transform(X_raw) if SCALER is not None else X_raw
    scores = MODEL.predict(X)

    return [
        {
            "index":           i,
            "predicted_score": round(min(max(float(s), 0), 100), 2),
            "grade":           grade(float(s)),
        }
        for i, s in enumerate(scores)
    ]