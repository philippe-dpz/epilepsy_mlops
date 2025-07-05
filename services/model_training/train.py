import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras

def load_data(base_path):
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    Y_train = np.load(os.path.join(base_path, "Y_train.npy"))
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    Y_test = np.load(os.path.join(base_path, "Y_test.npy"))
    return X_train, Y_train, X_test, Y_test

def main(args):
    # Define paths
    data_path = args.data_path
    model_path = args.model_path
    metrics_path = args.metrics_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    print("📂 Loading data...")
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    if len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, 178, 1)
        X_test = X_test.reshape(-1, 178, 1)

    print("🧠 Building model...")
    model = Sequential([
        LSTM(56, input_shape=(178, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(56),
        Dropout(0.3),
        Dense(20, activation='tanh'),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    with mlflow.start_run():
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("input_shape", (178, 1))
        mlflow.log_param("n_epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("🚀 Training...")
        hist = model.fit(
            X_train, Y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_test, Y_test),
            shuffle=False,
            callbacks=[early_stopping]
        )

        mlflow.log_metric("train_accuracy", float(hist.history["accuracy"][-1]))
        mlflow.log_metric("val_accuracy", float(hist.history["val_accuracy"][-1]))
        mlflow.log_metric("train_loss", float(hist.history["loss"][-1]))
        mlflow.log_metric("val_loss", float(hist.history["val_loss"][-1]))

        model.save(model_path)
        mlflow.keras.log_model(model, "epilepsy_model")

        metrics = {
            "train_accuracy": float(hist.history["accuracy"][-1]),
            "val_accuracy": float(hist.history["val_accuracy"][-1]),
            "train_loss": float(hist.history["loss"][-1]),
            "val_loss": float(hist.history["val_loss"][-1]),
            "epochs": len(hist.history["loss"]),
            "batch_size": args.batch_size,
            "architecture": "2xLSTM_+_Dense"
        }

        with open(metrics_path, "w") as f:
            json.dump({"model_path": model_path, "metrics": metrics}, f)

        print(f"✅ Training complete. Metrics saved in '{metrics_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data folder")
    parser.add_argument("--model_path", type=str, default="models/epilepsy_model.keras", help="Where to save the model")
    parser.add_argument("--metrics_path", type=str, default="metrics/model_metrics.json", help="Where to save the metrics JSON")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=15)
    args = parser.parse_args()

    main(args)
