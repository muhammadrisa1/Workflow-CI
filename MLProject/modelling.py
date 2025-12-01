import os
import random
import numpy as np
import pandas as pd
import argparse

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# LightGBM
from lightgbm import LGBMClassifier

# TensorFlow / Keras for autoencoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


print("\nLoad Dataset")

parser = argparse.ArgumentParser()
parser.add_argument("--train_final", type=str, required=True)
parser.add_argument("--test_final", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
args = parser.parse_args()


train_final = pd.read_csv(args.train_final)
test_final = pd.read_csv(args.test_final)

print("Dataset Loaded.")
print(f"Train shape : {train_final.shape}")
print(f"Test shape  : {test_final.shape}")

# Split features-labels
X_train = train_final.drop(columns=["Class"])
y_train = train_final["Class"]

X_test = test_final.drop(columns=["Class"])
y_test = test_final["Class"]

print("\nFeature split OK.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

# Only normal data
normal_data = X_train[y_train == 0]
print(f"Normal data for autoencoder: {normal_data.shape}")

# Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
normal_scaled = scaler.transform(normal_data)

n_features = X_train.shape[1]
encoding_dim = max(8, n_features // 4)


def build_autoencoder(input_dim, encoding_dim):
    input_layer = keras.Input(shape=(input_dim,))
    x = layers.Dense(max(encoding_dim * 4, 64), activation="relu")(input_layer)
    x = layers.Dense(encoding_dim * 2, activation="relu")(x)
    encoded = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(encoding_dim * 2, activation="relu")(encoded)
    x = layers.Dense(max(encoding_dim * 4, 64), activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="linear")(x)
    return keras.Model(inputs=input_layer, outputs=decoded)


# ---- MLflow RUN (satu-satunya!) ----
with mlflow.start_run():

    mlflow.autolog()
    mlflow.tensorflow.autolog()

    # Train Autoencoder
    print("\nTraining autoencoder...")
    autoencoder = build_autoencoder(n_features, encoding_dim)
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    autoencoder.fit(
        normal_scaled,
        normal_scaled,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early],
        verbose=1
    )

    # Reconstruction error
    recon_train = autoencoder.predict(X_train_scaled)
    recon_test = autoencoder.predict(X_test_scaled)

    recon_train_err = np.mean((recon_train - X_train_scaled) ** 2, axis=1)
    recon_test_err = np.mean((recon_test - X_test_scaled) ** 2, axis=1)

    # Feature enhancement
    X_train_enh = np.hstack([X_train_scaled, recon_train_err.reshape(-1, 1)])
    X_test_enh = np.hstack([X_test_scaled, recon_test_err.reshape(-1, 1)])

    print(f"Enhanced X_train: {X_train_enh.shape}")
    print(f"Enhanced X_test : {X_test_enh.shape}")

    # LightGBM Training
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=n_neg / n_pos
    )

    print("\nTraining LightGBM...")
    clf.fit(X_train_enh, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_enh)
    y_proba = clf.predict_proba(X_test_enh)[:, 1]

    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    })

    # Log confusion matrix as text
    cm = confusion_matrix(y_test, y_pred)
    cm_path = "confusion_matrix.txt"
    np.savetxt(cm_path, cm, fmt="%d")
    mlflow.log_artifact(cm_path)

    # Save LightGBM model
    mlflow.sklearn.log_model(clf, args.model_output)

print("\nModel completed.")
