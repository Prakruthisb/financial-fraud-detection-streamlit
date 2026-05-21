from datetime import datetime
from pathlib import Path
 
import pandas as pd
 
from evidently.report import Report
from evidently.metric_preset import (
    ClassificationPreset
)
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance
)

from app.monitoring.drift import get_column_mapping
 
# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 5. MODEL PERFORMANCE REPORT
# Is recall / precision degrading over time?
# Requires actual_fraud labels — only works for labelled prediction window.
# =============================================================================
 
def run_performance_report(
    current_df: pd.DataFrame,
    reference_path: str = REFERENCE_PATH,
    save_path: str      = None,
) -> dict:
    """
    Compare model performance on current window vs reference.
    current_df must include 'target' and 'prediction' columns.
    """
    if "target" not in current_df.columns or "prediction" not in current_df.columns:
        raise ValueError(
            "current_df must contain 'target' and 'prediction' columns. "
            "These are only available once ground-truth labels arrive."
        )
 
    reference_df = pd.read_parquet(reference_path)
    col_mapping  = get_column_mapping()
 
    report = Report(metrics=[
        ClassificationPreset(),
        ClassificationQualityMetric(),
        ClassificationClassBalance(),
    ])
 
    report.run(
        reference_data = reference_df,
        current_data   = current_df,
        column_mapping = col_mapping,
    )
 
    if save_path is None:
        ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        save_path = str(REPORTS_DIR / f"performance_{ts}.html")
    report.save_html(save_path)
    print(f"Performance report saved → {save_path}")
 
    # Extract recall and precision for alerting
    result      = report.as_dict()
    metrics_out = {"report_path": save_path}
 
    for m in result.get("metrics", []):
        if m.get("metric") == "ClassificationQualityMetric":
            cur = m.get("result", {}).get("current", {})
            metrics_out["recall"]    = cur.get("recall", None)
            metrics_out["precision"] = cur.get("precision", None)
            metrics_out["f1"]        = cur.get("f1", None)
 
    return metrics_out