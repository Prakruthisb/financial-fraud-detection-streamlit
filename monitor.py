import time

from app.monitoring.drift import prepare_current_window, run_drift_report
from app.monitoring.performance import run_performance_report
from app.monitoring.alerts import check_alerts

while True:
    print("🔄 Running monitoring...")

    try:
        df = prepare_current_window(days=1)

        drift = run_drift_report(df)

        perf = None
        if "target" in df.columns and df["target"].sum() > 0:
            perf = run_performance_report(df)

        alerts = check_alerts(drift, perf)

        for a in alerts:
            print(a)

    except Exception as e:
        print("Error:", e)

    time.sleep(300)  # every 5 minutes