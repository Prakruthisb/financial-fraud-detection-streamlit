import pandas as pd
import numpy as np
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are identifiers or cause leakage."""
    def __init__(self, cols):
        self.cols = cols
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])
 
 
class TimeFeatures(BaseEstimator, TransformerMixin):
    """Extract hour and day from the 'step' column, then drop step."""
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        X['hour'] = X['step'] % 24
        X['day']  = X['step'] // 24
        X = X.drop(columns=['step'])
        return X
 
 
class AmountFeatures(BaseEstimator, TransformerMixin):
    """
    Flag high-amount transactions and compute amount-to-balance ratio.
    The 90th-percentile threshold is LEARNED from training data only (fit),
    then applied to test data (transform) — no leakage.
    """
    def fit(self, X, y=None):
        self.amount_90th_ = X['amount'].quantile(0.90)
        return self
 
    def transform(self, X):
        X = X.copy()
        X['high_amount']            = (X['amount'] > self.amount_90th_).astype(int)
        X['amount_to_balance_ratio'] = X['amount'] / (X['oldbalanceOrg'] + 1)
        X['is_night']               = (X['hour'] < 6).astype(int)
        X['night_heist']            = (X['is_night'] & X['high_amount']).astype(int)
        X['zero_balance_before']    = (X['oldbalanceOrg'] == 0).astype(int)
        return X
 
 
class VelocityFeatures(BaseEstimator, TransformerMixin):
    """
    Compute hourly transaction velocity and amount deviation.
    Both are fitted on training data, then mapped to test data — no leakage.
    """
    def fit(self, X, y=None):
        # Learn hourly counts and averages from training data
        self.hourly_counts_ = X.groupby(['day', 'hour'])['amount'].count()
        self.hourly_avg_    = X.groupby('hour')['amount'].mean()
        return self
 
    def transform(self, X):
        X = X.copy()
        # Map learned stats — unknown (day, hour) combos get NaN → fill with median
        X['hourly_velocity'] = (
            X.set_index(['day', 'hour']).index.map(self.hourly_counts_)
        )
        X['hourly_velocity'] = X['hourly_velocity'].fillna(
            self.hourly_counts_.median()
        )
        hourly_avg_mapped    = X['hour'].map(self.hourly_avg_)
        X['amount_deviation'] = X['amount'] / hourly_avg_mapped
        return X
 
 
class LogTransform(BaseEstimator, TransformerMixin):
    """Apply log1p to skewed numerical columns."""
    def __init__(self, cols):
        self.cols = cols
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X
 
 
class OneHotEncodeType(BaseEstimator, TransformerMixin):
    """
    One-hot encode the 'type' column.
    Categories are LEARNED from training data to avoid unseen-category issues.
    """
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        dummies = pd.get_dummies(X['type'], prefix='type')
        #we should not use drop_first=True because we are using a tree-based model like XGBoost, which does not require it. 
        #Keeping all categories ensures consistency and avoids feature mismatch issues.
        self.columns_ = dummies.columns  # save all columns seen during training
        return self

    def transform(self, X):
        dummies = pd.get_dummies(X['type'], prefix='type')

        # Add missing columns
        for col in self.columns_:
            if col not in dummies:
                dummies[col] = 0

        # Ensure same order
        dummies = dummies[self.columns_]

        X = X.drop('type', axis=1)
        X = pd.concat([X, dummies], axis=1)

        return X