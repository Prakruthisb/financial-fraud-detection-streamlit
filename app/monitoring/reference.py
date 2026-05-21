from pathlib import Path
 
import pandas as pd
import joblib
 
# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 2. REFERENCE DATA BUILDER
# Build a reference snapshot from your training data.
# Run this ONCE after training — it becomes the baseline for all drift checks.
# =============================================================================
 
def build_reference_data(
    train_csv_path: str = "data/Fraud.csv",
    pipeline_path: str  = PIPELINE_PATH,
    output_path: str    = REFERENCE_PATH,
    sample_size: int    = 10_000,
):
    """
    Create reference data from the training set.
    Transforms raw data through the pipeline (minus the model step),
    adds fraud probabilities and labels, then saves as parquet.
 
    Run this once:
        python fraud_monitoring.py --build-reference
    """
    print("Building reference dataset...")
 
    pipeline = joblib.load(pipeline_path)
    df       = pd.read_csv(train_csv_path)
 
    # Use a stratified sample to keep it manageable
    fraud     = df[df["isFraud"] == 1].sample(min(500, len(df[df["isFraud"]==1])),
                                               random_state=42)
    legit     = df[df["isFraud"] == 0].sample(sample_size - len(fraud),
                                               random_state=42)
    sample    = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    y_ref     = sample["isFraud"].values
    X_ref     = sample.drop(columns=["isFraud"])
 
    # Get transformed features
    X_trans = X_ref.copy()
    for _, transformer in pipeline.steps[:-1]:
        X_trans = transformer.transform(X_trans)
 
    # Add prediction columns — Evidently needs these for performance monitoring
    probs              = pipeline.predict_proba(X_ref)[:, 1]
    X_trans["target"]  = y_ref
    X_trans["prediction"] = (probs > 0.9).astype(int)
    X_trans["fraud_probability"] = probs
 
    X_trans.to_parquet(output_path, index=False)
    print(f"Reference data saved → {output_path}")
    print(f"  Shape : {X_trans.shape}")
    print(f"  Fraud : {y_ref.mean()*100:.2f}%")
    return X_trans