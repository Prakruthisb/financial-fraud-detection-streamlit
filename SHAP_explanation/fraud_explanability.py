import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import streamlit as st

# =============================================================================
# SHAP Explainability for Fraud Detection
# Covers 4 use cases:
#   1. Global  — what features matter most overall
#   2. Local   — why THIS transaction was flagged
#   3. Summary — how each feature pushes predictions
#   4. Single  — clean text explanation for non-technical stakeholders
# =============================================================================


def get_explainer(pipeline):
    """
    Extract the XGBoost model from the pipeline and create a SHAP TreeExplainer.
    TreeExplainer is the correct explainer for XGBoost — fast and exact.
    """
    model = pipeline.named_steps['model']
    booster = model.get_booster()          # extract raw Booster object
    explainer = shap.TreeExplainer(booster)  # SHAP handles Booster better
    return explainer


def get_transformed_features(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run X through all pipeline steps EXCEPT the final model,
    so we get the feature matrix that XGBoost actually sees.
    SHAP operates on this transformed data — not the raw input.
    """
    # Get all steps except the last one ('model')
    transform_steps = pipeline.steps[:-1]

    X_transformed = X.copy()
    for _, transformer in transform_steps:
        X_transformed = transformer.transform(X_transformed)

    return X_transformed


# =============================================================================
# 1. GLOBAL EXPLANATION — feature importance across all test transactions
# =============================================================================

def plot_global_importance(pipeline, X_test: pd.DataFrame, max_display: int = 15):
    """
    Bar chart of mean absolute SHAP values.
    Answers: "Which features does the model rely on most overall?"
    """
    explainer    = get_explainer(pipeline)
    X_transformed = get_transformed_features(pipeline, X_test)
    shap_values  = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.title('Global Feature Importance (mean |SHAP value|)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('shap_global_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: shap_global_importance.png")


# =============================================================================
# 2. SUMMARY PLOT — direction + magnitude of every feature
# =============================================================================

def plot_summary(pipeline, X_test: pd.DataFrame, max_display: int = 15):
    """
    Beeswarm plot showing SHAP value distribution for each feature.
    Color = feature value (red = high, blue = low).
    Position = impact on prediction (right = pushes toward fraud).
    Answers: "High amount_to_balance_ratio pushes toward fraud — how strongly?"
    """
    explainer     = get_explainer(pipeline)
    X_transformed = get_transformed_features(pipeline, X_test)
    shap_values   = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary — Feature Impact on Fraud Prediction', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: shap_summary.png")


# =============================================================================
# 3. LOCAL EXPLANATION — why a single transaction was flagged
# =============================================================================

def explain_transaction(pipeline, raw_transaction: dict, threshold: float = 0.9):
    """
    Waterfall plot for a single transaction.
    Shows exactly which features pushed the prediction toward or away from fraud.
    Answers: "Why did the model flag THIS transaction?"

    Returns the SHAP values dict so you can use it in an API response too.
    """
    df            = pd.DataFrame([raw_transaction])
    explainer     = get_explainer(pipeline)
    X_transformed = get_transformed_features(pipeline, df)
    shap_values   = explainer.shap_values(X_transformed)

    # Fraud probability
    prob    = pipeline.predict_proba(df)[:, 1][0]
    verdict = "FRAUD" if prob > threshold else "LEGITIMATE"

    print(f"\nTransaction verdict : {verdict}")
    print(f"Fraud probability   : {prob:.4f}  (threshold = {threshold})")

    # Waterfall plot — most intuitive for a single prediction
    shap_explanation = shap.Explanation(
        values         = shap_values[0],
        base_values    = explainer.expected_value,
        data           = X_transformed.iloc[0].values,
        feature_names  = X_transformed.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, max_display=12, show=False)
    plt.title(f'Why this transaction was flagged as {verdict} (p={prob:.3f})',
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('shap_single_transaction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: shap_single_transaction.png")

    # Return structured explanation for API use
    feature_impacts = dict(zip(
        X_transformed.columns.tolist(),
        shap_values[0].tolist()
    ))
    # Sort by absolute impact, highest first
    feature_impacts = dict(
        sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    return {
        'fraud_probability' : round(float(prob), 4),
        'is_fraud'          : bool(prob > threshold),
        'base_rate'         : round(float(explainer.expected_value), 4),
        'feature_impacts'   : {k: round(v, 4) for k, v in feature_impacts.items()}
    }

def explain_transaction_streamlit(pipeline, df, threshold: float = 0.9):

    # st.write('function called')
    explainer     = get_explainer(pipeline)
    # st.write('transforming the features ')
    X_transformed = get_transformed_features(pipeline, df)
    # st.write("transformed the features")
    shap_values   = explainer.shap_values(X_transformed)

    # st.write(X_transformed.dtypes)
    # st.write(X_transformed.head())    

    # Prediction
    prob = pipeline.predict_proba(df)[:, 1][0]
    verdict = "FRAUD" if prob > threshold else "LEGITIMATE"

    # 🟢 Show results in Streamlit
    st.subheader("Prediction Result")
    st.write(f"**Verdict:** {verdict}")
    st.write(f"**Fraud Probability:** {prob:.4f}")

    # 🔥 Handle SHAP values properly
    shap_value = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]

    # SHAP explanation
    shap_explanation = shap.Explanation(
        values=shap_value,
        base_values=expected_value,
        data=X_transformed.iloc[0],
        feature_names=X_transformed.columns.tolist()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, max_display=12, show=False)
    st.pyplot(fig)

    # Return structured output (optional)
    feature_impacts = dict(zip(
        X_transformed.columns.tolist(),
        shap_value.tolist()
    ))

    feature_impacts = dict(
        sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        'fraud_probability': round(float(prob), 4),
        'is_fraud': bool(prob > threshold),
        'feature_impacts': {k: round(v, 4) for k, v in feature_impacts.items()}
    }

# =============================================================================
# 4. PLAIN ENGLISH EXPLANATION — for non-technical stakeholders / API response
# =============================================================================

def explain_in_plain_english(explanation: dict, top_n: int = 3) -> str:
    """
    Convert the SHAP output from explain_transaction() into a human-readable
    explanation. Useful for fraud analyst dashboards or API responses.

    Example output:
        "This transaction was flagged as likely fraud (93.2% confidence).
         The top reasons are:
           → amount_to_balance_ratio was very high, strongly increasing fraud risk
           → type_TRANSFER indicates a transfer transaction, increasing fraud risk
           → hour was 3 (nighttime), slightly increasing fraud risk"
    """
    prob    = explanation['fraud_probability']
    impacts = explanation['feature_impacts']

    # Separate fraud-pushing (positive) from safe-pushing (negative) features
    fraud_pushers = {k: v for k, v in impacts.items() if v > 0}
    safe_pushers  = {k: v for k, v in impacts.items() if v < 0}

    top_fraud = list(fraud_pushers.items())[:top_n]
    top_safe  = list(safe_pushers.items())[:top_n]

    def strength_label(val):
        a = abs(val)
        if a > 0.5:   return "strongly"
        if a > 0.2:   return "moderately"
        if a > 0.05:  return "slightly"
        return "marginally"

    verdict = "FRAUD" if explanation['is_fraud'] else "LEGITIMATE"
    lines   = [
        f"This transaction was flagged as {verdict} "
        f"({prob*100:.1f}% fraud probability).\n"
    ]

    if top_fraud:
        lines.append("Factors increasing fraud risk:")
        for feat, val in top_fraud:
            lines.append(f"  → {feat}: {strength_label(val)} increases fraud risk")

    if top_safe:
        lines.append("\nFactors decreasing fraud risk:")
        for feat, val in top_safe:
            lines.append(f"  → {feat}: {strength_label(val)} decreases fraud risk")

    return "\n".join(lines)


# =============================================================================
# 5. FORCE PLOT — interactive HTML (great for notebooks)
# =============================================================================

def force_plot_html(pipeline, raw_transaction: dict, save_path='shap_force.html'):
    """
    Interactive force plot — best viewed in a Jupyter notebook.
    Shows base value + each feature's push toward/away from fraud.
    Saves as standalone HTML you can open in a browser or embed in a dashboard.
    """
    shap.initjs()

    df            = pd.DataFrame([raw_transaction])
    explainer     = get_explainer(pipeline)
    X_transformed = get_transformed_features(pipeline, df)
    shap_values   = explainer.shap_values(X_transformed)

    force = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_transformed.iloc[0],
        feature_names=X_transformed.columns.tolist(),
        matplotlib=False
    )
    shap.save_html(save_path, force)
    print(f"Interactive force plot saved to {save_path}")
    return force   # renders inline in Jupyter


# =============================================================================
# USAGE
# =============================================================================

if __name__ == '__main__':

    # Load your saved pipeline
    pipeline = joblib.load('models/fraud_pipeline.pkl')

    # Load test data (use the same test split you used during training)
    df     = pd.read_csv('data/Fraud.csv')
    df = df.sample(100000,random_state=42)
    X      = df.drop(columns=['isFraud'])
    y      = df['isFraud']

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Global plots (run once, tells you how the model thinks overall) ──────
    plot_global_importance(pipeline, X_test)
    plot_summary(pipeline, X_test)

    # ── Local explanation (run per transaction) ──────────────────────────────
    suspicious_transaction = {
        'step'          : 3,
        'type'          : 'TRANSFER',
        'amount'        : 450000.0,
        'nameOrig'      : 'C1305486145',
        'oldbalanceOrg' : 450000.0,
        'nameDest'      : 'C553264065',
        'oldbalanceDest': 0.0,
    }

    explanation = explain_transaction(pipeline, suspicious_transaction, threshold=0.9)
    print("\nStructured explanation (for API response):")
    print(explanation)

    print("\nPlain English explanation (for analyst dashboard):")
    print(explain_in_plain_english(explanation))

    # ── Interactive HTML force plot (best in Jupyter) ────────────────────────
    force_plot_html(pipeline, suspicious_transaction)