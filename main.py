import pandas as pd
from src.training.train import train, save_pipeline
from src.inference.predict import load_pipeline, predict
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('data/Fraud.csv')
 
    # Train and tune
    pipeline = train(df, threshold=0.9)
 
    # Save the full pipeline (transforms + model)
    save_pipeline(pipeline, 'fraud_pipeline.pkl')
 
    # Load and predict on a raw transaction
    pipeline = load_pipeline('fraud_pipeline.pkl')
    # print(pipeline)
 
    sample_transaction = {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 181.0,
        'nameOrig': 'C1305486145',
        'oldbalanceOrg': 181.0,
        'newbalanceOrig': 0.0,
        'nameDest': 'C553264065',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFlaggedFraud': 0
    }
 
    result = predict(pipeline, sample_transaction, threshold=0.9)
    print(result)


    # # PR(Precision Recall) Curve
    # df = pd.read_csv('data/Fraud.csv')
    # X = df.drop(['isFraud'],axis=1)
    # y = df['isFraud']

    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # pipeline = load_pipeline('models/fraud_pipeline.pkl')
    # y_probs = pipeline.predict_proba(X_test)[:, 1]

    # # Get precision, recall, thresholds
    # precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    # # Plot PR Curve
    # plt.figure(figsize=(8,6))
    # plt.plot(recall, precision, label="PR Curve")

    # # Mark your chosen threshold = 0.9
    # threshold = 0.9

    # # Find closest threshold index
    # idx = np.argmin(np.abs(thresholds - threshold))

    # plt.scatter(recall[idx], precision[idx], color='red', label=f"Threshold = {threshold}")

    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.legend()
    # plt.grid()

    # plt.savefig("pr_curve.png")
    # plt.show()