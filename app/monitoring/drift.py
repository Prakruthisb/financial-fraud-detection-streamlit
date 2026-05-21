from datetime import datetime, timezone
from pathlib import Path
 
import pandas as pd
import joblib
import os

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset
)
from evidently.metrics import (
    DatasetMissingValuesMetric,
    ColumnDriftMetric
)
 
from app.monitoring.logger import load_predictions
 
# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = os.getenv("DATABASE_URL")
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORT_DIR = "/tmp/monitoring_reports"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# REPORT_DIR = os.path.join(BASE_DIR, "monitoring_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def get_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target          = "target",
        prediction      = "prediction",
        numerical_features = [
            "amount", "oldbalanceOrg", "oldbalanceDest",
            "hour", "day", "amount_to_balance_ratio",
            "hourly_velocity", "amount_deviation",
            "high_amount", "is_night", "night_heist",
            "zero_balance_before",
        ],
        categorical_features = [
            "type_CASH_OUT", "type_DEBIT",
            "type_PAYMENT", "type_TRANSFER",
        ],
    )

def run_drift_report(
    current_df: pd.DataFrame,
    reference_path: str = REFERENCE_PATH,
    save_path: str      = None,
) -> dict:
    """
    Compare current window vs reference data.
    Returns a dict of drift metrics + saves HTML report.
 
    current_df must have the same columns as reference (transformed features).
    """
    if not Path(reference_path).exists():
        raise FileNotFoundError(
            f"Reference data not found at {reference_path}. "
            "Run build_reference_data() first."
        )
    import streamlit as st
    # st.write("getting reference data")
    reference_df = pd.read_parquet(reference_path)
    # st.write('getting column mapping')
    col_mapping  = get_column_mapping()
    # st.write('preparing report')
    # st.write(col_mapping)
 
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name="amount"),
        ColumnDriftMetric(column_name="amount_to_balance_ratio"),
        ColumnDriftMetric(column_name="hourly_velocity"),
    ])
    
    #debug
    # st.write("Current columns:", current_df.columns)
    # st.write("Reference columns:", reference_df.columns)

    # if "target" in current_df.columns:
        # st.write("Current target nulls:", current_df["target"].isna().sum())
        # st.write("Current rows:", len(current_df))

    # if "target" in reference_df.columns:
        # st.write("Reference target nulls:", reference_df["target"].isna().sum())

    # if "target" in reference_df.columns and "target" not in current_df.columns:
    #     st.write("⚠️ Dropping target from reference to match current data")
    #     reference_df = reference_df.drop(columns=["target"])

    col_mapping.target = None

    report.run(
        reference_data = reference_df,
        current_data   = current_df,
        column_mapping = col_mapping,
    )
 
    # Save HTML report
    if save_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        # st.write(ts)
        # save_path = str(REPORT_DIR / f"drift_{ts}.html")
        save_path = os.path.join(REPORT_DIR, f"drift_{ts}.html")
        # st.write(f"Saving drift report to {save_path}...")
    report.save_html(save_path)
    # st.write(f"Drift report saved → {save_path}")
 
    # Extract key metrics as dict for alerting
    result     = report.as_dict()
    metrics_out = {}
 
    for m in result.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            r = m.get("result", {})
            metrics_out["drift_detected"]       = r.get("dataset_drift", False)
            metrics_out["drifted_features"]     = r.get("number_of_drifted_columns", 0)
            metrics_out["total_features"]       = r.get("number_of_columns", 0)
            metrics_out["drift_share"]          = r.get("share_of_drifted_columns", 0.0)
        if m.get("metric") == "DatasetMissingValuesMetric":
            r = m.get("result", {})
            metrics_out["missing_share"] = r.get("current", {}).get(
                "share_of_missing_values", 0.0
            )
 
    metrics_out["report_path"] = save_path
    return metrics_out

def prepare_current_window(
    days: int           = 7,
    pipeline_path: str  = PIPELINE_PATH,
    db_path: str        = DB_PATH,
    threshold: float    = 0.9,
) -> pd.DataFrame:
    """
    Load recent predictions from DB, re-transform through pipeline,
    and add target/prediction columns for Evidently.
    """
    pred_df = load_predictions(days=days)
 
    if pred_df.empty:
        raise ValueError(f"No predictions found in the last {days} days.")
 
    pipeline = joblib.load(pipeline_path)
    # print(pred_df.columns)
    # Reconstruct raw transaction format from logged columns
    raw = pd.DataFrame({
        "step"          : pred_df["step"].fillna(1).astype(int),
        "type"          : pred_df["type"].fillna("TRANSFER"),
        "amount"        : pred_df["amount"].fillna(0),
        "nameOrig"      : "C_LOGGED",
        "oldbalanceOrg" : pred_df["oldbalanceorg"].fillna(0),
        "nameDest"      : "C_DEST",
        "oldbalanceDest": pred_df["oldbalancedest"].fillna(0),
    })
 
    X_trans = raw.copy()
    for _, transformer in pipeline.steps[:-1]:
        X_trans = transformer.transform(X_trans)
 
    X_trans["fraud_probability"] = pred_df["fraud_probability"].values
    X_trans["prediction"]        = (
        pred_df["fraud_probability"] > threshold
    ).astype(int).values
 
    # Add actual labels if available (needed for performance report)
    pred_df = pred_df.reset_index(drop=True)
    X_trans.head(10)
    X_trans = X_trans.reset_index(drop=True)

    if "actual_fraud" in pred_df.columns:
        labeled_mask = pred_df["actual_fraud"].notna()

        if labeled_mask.sum() > 0:
            X_trans = X_trans.loc[labeled_mask].copy()
            X_trans["target"] = pred_df.loc[labeled_mask, "actual_fraud"].values
 
    return X_trans