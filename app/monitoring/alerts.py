from pathlib import Path
 
# Alert thresholds — tune these to your business tolerance
ALERT_THRESHOLDS = {
    "recall_min"         : 0.85,   # alert if recall drops below this
    "precision_min"      : 0.30,   # alert if precision drops below this
    "fraud_rate_max"     : 0.05,   # alert if live fraud rate exceeds 5%
    "drift_share_max"    : 0.30,   # alert if >30% of features are drifting
    "missing_values_max" : 0.01,   # alert if >1% of values are missing
}

# =============================================================================
# 6. ALERT SYSTEM
# Checks metrics against thresholds and prints warnings.
# In production: replace print() with Slack/email/PagerDuty notification.
# =============================================================================
 
def check_alerts(drift_metrics: dict, perf_metrics: dict = None) -> list[str]:
    """
    Returns a list of alert messages.
    Empty list = all clear.
    """
    alerts = []
 
    # Drift alerts
    drift_share = drift_metrics.get("drift_share", 0)
    if drift_share > ALERT_THRESHOLDS["drift_share_max"]:
        alerts.append(
            f"⚠️  DATA DRIFT: {drift_share*100:.0f}% of features are drifting "
            f"(threshold: {ALERT_THRESHOLDS['drift_share_max']*100:.0f}%). "
            "Consider retraining."
        )
 
    missing = drift_metrics.get("missing_share", 0)
    if missing > ALERT_THRESHOLDS["missing_values_max"]:
        alerts.append(
            f"⚠️  DATA QUALITY: {missing*100:.2f}% missing values detected "
            f"(threshold: {ALERT_THRESHOLDS['missing_values_max']*100:.2f}%). "
            "Check upstream data pipeline."
        )
 
    # Performance alerts (only if labels available)
    if perf_metrics:
        recall = perf_metrics.get("recall")
        if recall and recall < ALERT_THRESHOLDS["recall_min"]:
            alerts.append(
                f"🚨 PERFORMANCE DEGRADATION: Recall dropped to {recall:.2%} "
                f"(minimum: {ALERT_THRESHOLDS['recall_min']:.2%}). "
                "Urgent — model is missing fraud cases."
            )
 
        precision = perf_metrics.get("precision")
        if precision and precision < ALERT_THRESHOLDS["precision_min"]:
            alerts.append(
                f"⚠️  PERFORMANCE: Precision dropped to {precision:.2%} "
                f"(minimum: {ALERT_THRESHOLDS['precision_min']:.2%}). "
                "Too many false alarms — analysts may be overwhelmed."
            )
 
    if not alerts:
        alerts.append("✅ All metrics within acceptable thresholds.")
 
    return alerts

def fraud_rate_alert(pred_df):
    rate = pred_df["predicted_fraud"].mean()

    if rate > 0.05:
        return f"🚨 Fraud rate spike: {rate:.2%}"

    return None

