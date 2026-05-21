from fastapi import FastAPI
from src.inference.predict import load_pipeline
from src.pydantic_model.models import Transaction
import pandas as pd
import os

from app.monitoring.logger import log_prediction

app = FastAPI()

# Load pipeline once when API starts
# pipeline = load_pipeline('fraud_pipeline.pkl')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pipeline = load_pipeline(os.path.join(BASE_DIR, 'fraud_pipeline.pkl'))


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict_api(transaction: Transaction):
    try:
        data = transaction.model_dump()
        df = pd.DataFrame([data])

        prob = pipeline.predict_proba(df)[0][1]
        pred = prob > 0.9

        log_prediction(
            raw_transaction=data,
            fraud_probability=float(prob),
            predicted_fraud=bool(pred),
            threshold=0.9
        )

        return {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": bool(pred),
            "message": "Fraud detected" if pred else "Legitimate transaction"
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/health")
def health():
    return {"status": "ok"}