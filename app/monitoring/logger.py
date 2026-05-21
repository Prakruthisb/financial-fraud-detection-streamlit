import os
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = os.getenv("DATABASE_URL")   # from Render
# REFERENCE_PATH = os.path.join(BASE_DIR, "reference_data.parquet")
# PIPELINE_PATH  = os.path.join(BASE_DIR, "fraud_pipeline.pkl")
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── DB CONNECTION ─────────────────────────────────────────────────────────────
def get_connection():
    return create_engine(DATABASE_URL)

# ── INIT DB ───────────────────────────────────────────────────────────────────
def init_db():
    engine = get_connection()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                step INTEGER,
                type TEXT,
                amount FLOAT,
                oldbalanceOrg FLOAT,
                oldbalanceDest FLOAT,
                hour INTEGER,
                day INTEGER,
                fraud_probability FLOAT,
                predicted_fraud INTEGER,
                actual_fraud INTEGER,
                threshold FLOAT
            )
        """))

# ── LOG PREDICTION ────────────────────────────────────────────────────────────
def log_prediction(
    raw_transaction: dict,
    fraud_probability: float,
    predicted_fraud: bool,
    threshold: float,
    actual_fraud: int = None,
):
    init_db()
    engine = get_connection()

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predictions
            (timestamp, step, type, amount, oldbalanceOrg, oldbalanceDest,
             hour, day, fraud_probability, predicted_fraud, actual_fraud, threshold)
            VALUES (:timestamp, :step, :type, :amount, :oldbalanceOrg, :oldbalanceDest,
                    :hour, :day, :fraud_probability, :predicted_fraud, :actual_fraud, :threshold)
        """), {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": raw_transaction.get("step"),
            "type": raw_transaction.get("type"),
            "amount": raw_transaction.get("amount"),
            "oldbalanceOrg": raw_transaction.get("oldbalanceOrg"),
            "oldbalanceDest": raw_transaction.get("oldbalanceDest"),
            "hour": int(raw_transaction.get("step", 0)) % 24,
            "day": int(raw_transaction.get("step", 0)) // 24,
            "fraud_probability": round(float(fraud_probability), 6),
            "predicted_fraud": int(predicted_fraud),
            "actual_fraud": actual_fraud,
            "threshold": float(threshold),
        })

# ── UPDATE LABEL ──────────────────────────────────────────────────────────────
def update_actual_label(prediction_id: int, actual_fraud: int):
    engine = get_connection()

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE predictions SET actual_fraud = :actual WHERE id = :id",
            {"actual": actual_fraud, "id": prediction_id}
        ))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_predictions(days: int = 7) -> pd.DataFrame:
    init_db()

    engine = get_connection()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    query = """
        SELECT * FROM predictions
        WHERE timestamp >= %s
        ORDER BY timestamp
    """

    df = pd.read_sql(query, engine, params=(since,))

    return df