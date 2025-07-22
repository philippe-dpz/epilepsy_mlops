import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow

def load_data(base_path):
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    Y_train = np.load(os.path.join(base_path, "Y_train.npy"))
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    Y_test = np.load(os.path.join(base_path, "Y_test.npy"))
    return X_train, Y_train, X_test, Y_test

def main(args):
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("epilepsy_training")

    mlflow.tensorflow.autolog()

    X_train, Y_train, X_test, Y_test = load_data(args.data_path)

    if len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, 178, 1)
        X_test = X_test.reshape(-1, 178, 1)

    model = Sequential([
        Input(shape=(178, 1)),
        LSTM(56, return_sequences=True),
        Dropout(0.3),
        LSTM(56),
        Dropout(0.3),
        Dense(20, activation='tanh'),
        Dense(2, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with mlflow.start_run():
        mlflow.log_param("architecture", "2xLSTM+Dense")
        mlflow.log_param("input_shape", str((178, 1)))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        history = model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stopping],
            shuffle=False
        )

        mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("final_val_accuracy", history.history["val_accuracy"][-1])
        mlflow.log_metric("final_train_loss", history.history["loss"][-1])
        mlflow.log_metric("final_val_loss", history.history["val_loss"][-1])

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        print(f"Saving model to {args.model_path}")
        model.save(args.model_path)

        mlflow.keras.log_model(model)

        mlflow.log_artifact(args.model_path, artifact_path="keras_model_file")

        metrics = {
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "epochs": len(history.history["loss"]),
            "batch_size": args.batch_size,
        }

        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
        with open(args.metrics_path, "w") as f:
            json.dump(metrics, f)

        mlflow.log_artifact(args.metrics_path, artifact_path="metrics")

        print("Training complete and artifacts logged to MLflow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/app/data/processed")
    parser.add_argument("--model_path", type=str, default="/app/models/epilepsy_model.keras")
    parser.add_argument("--metrics_path", type=str, default="/app/metrics/model_metrics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=15)
    args = parser.parse_args()
    main(args)
