from .transformers import DropColumns, TimeFeatures, OneHotEncodeType, AmountFeatures, VelocityFeatures, LogTransform
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier 

def build_pipeline(scale_pos_weight):
    pipeline = Pipeline(steps=[
        # 1. Drop identifier columns (high cardinality, no predictive value)
        ('drop_ids',        DropColumns(['nameOrig', 'nameDest', 'isFlaggedFraud'])),
 
        # 2. Extract time features from 'step'
        ('time_features',   TimeFeatures()),
 
        # 3. Encode transaction type
        ('encode_type',     OneHotEncodeType()),
 
        # 4. Drop post-transaction balances (cause leakage — unknowable at prediction time)
        ('drop_leaky',      DropColumns(['newbalanceOrig', 'newbalanceDest'])),
 
        # 5. Amount and balance features
        ('amount_features', AmountFeatures()),
 
        # 6. Velocity features (fitted on train, mapped to test)
        ('velocity',        VelocityFeatures()),
 
        # 7. Log transform skewed columns
        ('log_transform',   LogTransform(['amount', 'oldbalanceOrg', 'oldbalanceDest'])),
 
        # 8. XGBoost classifier
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            tree_method='hist',
        ))
    ])
    return pipeline