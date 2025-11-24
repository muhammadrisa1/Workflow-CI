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

def load_dataset(train_x, train_y, test_x, test_y):
    print("Loading datasets...")

    X_train_final = pd.read_csv(train_x)
    y_train_bal = pd.read_csv(train_y)
    X_test_final = pd.read_csv(test_x)
    y_test = pd.read_csv(test_y)

    # Jika dataframe memiliki 1 kolom, ubah ke Series
    if isinstance(y_train_bal, pd.DataFrame):
        y_train_bal = y_train_bal.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"X_train_final: {X_train_final.shape}")
    print(f"y_train_bal: {y_train_bal.shape}")
    print(f"X_test_final: {X_test_final.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train_final, y_train_bal, X_test_final, y_test


def train_model(train_x, train_y, test_x, test_y, model_output):
    
    X_train_final, y_train_bal, X_test_final, y_test = load_dataset(
        train_x, train_y, test_x, test_y
    )

    mlflow.set_experiment("Fraud_Detection_IsolationForest")

    with mlflow.start_run(run_name="IsolationForest_Final_Optimal"):
        mlflow.autolog()

        # Gunakan hanya data NORMAL untuk training
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

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(y_test, y_pred_final)
        recall = recall_score(y_test, y_pred_final)
        f1 = f1_score(y_test, y_pred_final)
        auc_score = roc_auc_score(y_test, -final_model.decision_function(X_test_final))
        cm = confusion_matrix(y_test, y_pred_final)
        tn, fp, fn, tp = cm.ravel()

        # Log params
        mlflow.log_param("contamination", 0.001)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_samples", len(normal_data))

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc_score)
        mlflow.log_metric("true_positives", tp)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_negatives", fn)

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        mlflow.log_metric("false_positive_rate", false_positive_rate)

        # Tags
        mlflow.set_tag("status", "production_ready")
        mlflow.set_tag("tuning_result", "optimal")
        mlflow.set_tag("model_type", "IsolationForest")
        mlflow.set_tag("project", "fraud_detection")
        mlflow.set_tag("data_source", "processed_creditcard")

        # Save model
        mlflow.sklearn.save_model(final_model, model_output)

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