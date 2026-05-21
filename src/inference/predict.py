import pandas as pd
import joblib

def load_pipeline(path='fraud_pipeline.pkl'):
    return joblib.load(path)
 
 
# =============================================================================
# STEP 5 — Predict on a raw transaction (as it would arrive in production)
# =============================================================================
 
def predict(pipeline, raw_transaction: dict, threshold: float = 0.9) -> dict:
    """
    Accept a raw transaction dict (exactly as it comes from the source system),
    run it through the full pipeline, return fraud probability and verdict.
 
    Example input:
        {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 181.0,
            'nameOrig': 'C1305486145',
            'oldbalanceOrg': 181.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'C553264065',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'isFlaggedFraud': 0
        }
    """
    df = pd.DataFrame([raw_transaction])
    prob = pipeline.predict_proba(df)[:, 1][0]
    return {
        'fraud_probability': round(float(prob), 4),
        'is_fraud':          bool(prob > threshold),
        'threshold_used':    threshold,
    }