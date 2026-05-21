import pandas as pd
import joblib

from src.pipeline.pipeline import build_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def train(df: pd.DataFrame, threshold: float = 0.9):
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
 
    # Stratified split preserves the 0.13% fraud rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
 
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    pipeline = build_pipeline(scale_pos_weight)
 
    # Hyperparameter search — only tuning XGBoost params (prefix: 'model__')
    param_grid = {
        'model__n_estimators':     [100, 200, 300],
        'model__max_depth':        [3, 5, 7],
        'model__learning_rate':    [0.01, 0.1, 0.2],
        'model__subsample':        [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    }
 
    # StratifiedKFold ensures minority class present in every fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
 
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        scoring='recall',   # maximize recall = minimize false negatives
        verbose=1,
        n_jobs=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
 
    best_pipeline = search.best_estimator_
    print(f"Best params: {search.best_params_}")
 
    # Evaluate with chosen threshold
    y_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_probs > threshold).astype(int)
 
    print(f"\n--- Evaluation at threshold = {threshold} ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_probs):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_probs):.4f}")
    print(f"Recall  : {recall_score(y_test, y_pred):.4f}  ← minimize false negatives")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
 
    return best_pipeline
 
 
# =============================================================================
# STEP 4 — Save & Load the FULL pipeline (not just the model)
# =============================================================================
 
def save_pipeline(pipeline, path='fraud_pipeline.pkl'):
    """Save the entire pipeline — transforms + model together."""
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")
 
 
