import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import requests
import os
from urllib.parse import urlparse

def download_file(url, local_filename):
    """Download file from URL if it's a web URL"""
    parsed = urlparse(url)
    if parsed.scheme in ('http', 'https'):
        print(f"Downloading {url} to {local_filename}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"Download completed: {local_filename}")
        return local_filename
    else:
        
        return url

def load_dataset(train_x, train_y, test_x, test_y):
    print("Loading datasets...")

    # Download files dengan nama yang sesuai dengan URL asli
    local_train_x = download_file(train_x, "creditcard_train_x.csv")
    local_train_y = download_file(train_y, "creditcard_train_y.csv") 
    local_test_x = download_file(test_x, "creditcard_test_x.csv")     
    local_test_y = download_file(test_y, "creditcard_test_y.csv")     

    
    X_train_final = pd.read_csv(local_train_x)
    y_train_bal = pd.read_csv(local_train_y)
    X_test_final = pd.read_csv(local_test_x)
    y_test = pd.read_csv(local_test_y)

    
    if isinstance(y_train_bal, pd.DataFrame) and y_train_bal.shape[1] == 1:
        y_train_bal = y_train_bal.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    print(f"X_train_final shape: {X_train_final.shape}")
    print(f"y_train_bal shape: {y_train_bal.shape}")
    print(f"X_test_final shape: {X_test_final.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train_final, y_train_bal, X_test_final, y_test

def train_model(train_x, train_y, test_x, test_y, model_output):
    
    X_train_final, y_train_bal, X_test_final, y_test = load_dataset(
        train_x, train_y, test_x, test_y
    )

   
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    with mlflow.start_run(run_name="IsolationForest_Final_Optimal") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        normal_data = X_train_final[y_train_bal == 0]
        print(f"Normal data for training: {normal_data.shape}")

        final_model = IsolationForest(
            contamination=0.001,
            n_estimators=200,
            random_state=42
        )

        print("Training Isolation Forest model...")
        final_model.fit(normal_data)

        print("Making predictions...")
        final_predictions = final_model.predict(X_test_final)
        y_pred_final = np.where(final_predictions == -1, 1, 0)

       
        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(y_test, y_pred_final)
        recall = recall_score(y_test, y_pred_final)
        f1 = f1_score(y_test, y_pred_final)
        
        
        decision_scores = final_model.decision_function(X_test_final)
        auc_score = roc_auc_score(y_test, -decision_scores) 
        
        cm = confusion_matrix(y_test, y_pred_final)
        tn, fp, fn, tp = cm.ravel()

        
        mlflow.log_params({
            "contamination": 0.001,
            "n_estimators": 200,
            "random_state": 42,
            "training_samples": len(normal_data),
            "model_type": "IsolationForest"
        })

       
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc_score,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        })

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        mlflow.log_metric("false_positive_rate", false_positive_rate)

        
        mlflow.set_tags({
            "status": "production_ready",
            "tuning_result": "optimal", 
            "model_type": "IsolationForest",
            "project": "fraud_detection",
            "data_source": "processed_creditcard"
        })

        
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",  # Path dalam run MLflow
            registered_model_name="fraud-detection-isolation-forest"
        )
        
       
        import pickle
        os.makedirs(model_output, exist_ok=True)
        with open(os.path.join(model_output, "model.pkl"), "wb") as f:
            pickle.dump(final_model, f)
        
        print(f"Model saved to {model_output}/model.pkl")

        print("\nFINAL MODEL RESULTS")
        print(f"\nCONFUSION MATRIX:")
        print(f"                Predicted")
        print(f"               Normal   Fraud")
        print(f"Actual Normal   {tn:6}   {fp:6}")
        print(f"Actual Fraud    {fn:6}   {tp:6}")

        print(f"\nPERFORMANCE METRICS:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc_score:.4f}")

        print("\nModelling completed successfully!")
        print(f"MLflow Run: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_x", type=str, required=True)
    parser.add_argument("--train_y", type=str, required=True)
    parser.add_argument("--test_x", type=str, required=True)
    parser.add_argument("--test_y", type=str, required=True)
    parser.add_argument("--model_output", type=str, default="model")

    args = parser.parse_args()

    train_model(
        args.train_x,
        args.train_y,
        args.test_x,
        args.test_y,
        args.model_output
    )
