# Student Score Prediction — MLOps Workflow

Predict student exam scores based on **study hours**, **sleep hours**, and **attendance** using a full MLOps stack.

## Stack

| Concern | Tool |
|---|---|
| Code versioning | Git + GitHub |
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Pipeline orchestration | Prefect |
| Model serving | FastAPI |

## Dataset

**Students Performance Dataset** — Kaggle  
URL: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression  
File: `student_performance.csv` (~10 000 rows)  
Target: `Performance Index` (0–100)

Features used:
- `Hours Studied` — daily study hours
- `Sleep Hours` — nightly sleep hours
- `Attendance` — attendance percentage

## Project Structure

```
student_score_mlops/
├── data/
│   ├── raw/               # Original CSV (tracked by DVC)
│   └── processed/         # Train/test splits
├── models/                # Saved model artifacts
├── src/
│   ├── pipeline/
│   │   ├── ingest.py      # Download & validate data
│   │   ├── preprocess.py  # Feature engineering & splitting
│   │   ├── train.py       # Model training + MLflow logging
│   │   └── evaluate.py    # Evaluation metrics
│   └── api/
│       └── main.py        # FastAPI prediction server
├── flows/
│   └── training_flow.py   # Prefect orchestration flow
├── tests/
│   └── test_pipeline.py
├── dvc.yaml               # DVC pipeline stages
├── params.yaml            # Hyperparameters
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml         # GitHub Actions CI
```

## Quickstart

```bash
# 1. Clone and install
git clone <your-repo>
cd student_score_mlops
pip install -r requirements.txt

# 2. Pull data with DVC
dvc pull

# 3. Run the full Prefect training flow
python flows/training_flow.py

# 4. Launch MLflow UI
mlflow ui --port 5000

# 5. Serve predictions
uvicorn src.api.main:app --reload --port 8000

# 6. Predict (example)
 Invoke-RestMethod -Uri "http://localhost:8000/predict" `
-Method POST `
-Headers @{ "Content-Type" = "application/json" } `
-Body '{"study_hours": 3, "sleep_hours": 6, "attendance": 50}'
```

## Models Evaluated

- **LinearRegression** — baseline
- **Ridge** — L2 regularized
- **RandomForestRegressor** — ensemble
- **XGBRegressor** — gradient boosting (usually best)

Best model is registered automatically in MLflow Model Registry under `StudentScoreModel`.
