import pandas as pd
import joblib
import argparse

from app.monitoring.reference import build_reference_data
from app.monitoring.logger import init_db
from app.monitoring.drift import prepare_current_window, run_drift_report
from app.monitoring.performance import run_performance_report
from app.monitoring.alerts import check_alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument('--seed-demo',action='store_true')
    parser.add_argument("--run-reports", action="store_true")

    args = parser.parse_args()

    if args.build_reference:
        print("🔧 Building reference data...")
        build_reference_data()
        print("✅ Done!")

    if args.run_reports:
        print("📊 Running monitoring reports...")
        
        df = prepare_current_window()

        drift = run_drift_report(df)
        # perf = run_performance_report(df)

        if "target" in df.columns and df["target"].sum() > 0:
            perf = run_performance_report(df)
        else:
            print("⚠️ Skipping performance report (no fraud cases)")
            perf = None

        check_alerts(drift, perf)

        print("✅ Reports generated!")
    
    if args.seed_demo:
        print("🌱 Seeding demo data...")

        pipeline = joblib.load("fraud_pipeline.pkl")
        df = pd.read_csv("data/Fraud.csv").sample(200, random_state=42)

        from app.monitoring.logger import log_prediction

        for _, row in df.iterrows():
            raw = {
                "step": int(row["step"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
            }

            prob = float(pipeline.predict_proba(pd.DataFrame([raw]))[:, 1][0])
            is_fraud = prob > 0.9

            log_prediction(
                raw,
                prob,
                is_fraud,
                threshold=0.9,
                actual_fraud=int(row["isFraud"])
            )

        print("✅ Demo data inserted into DB")

    if not any([args.build_reference, args.run_reports, args.seed_demo]):
        print("⚠️ Please provide an argument:")
        print("--build-reference | --seed-demo | --run-reports")

if __name__ == "__main__":
    init_db()
    main()