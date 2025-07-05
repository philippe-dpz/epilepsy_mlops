import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
import mlflow
import mlflow.keras

# ✅ Absolute paths for local Windows setup
PROCESSED_X_TRAIN_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\X_train.npy"
PROCESSED_Y_TRAIN_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\Y_train.npy"
PROCESSED_X_TEST_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\X_test.npy"
PROCESSED_Y_TEST_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\Y_test.npy"
MODEL_SAVE_PATH = r"C:\Users\phili\epilepsy_mlops\models\epilepsy_model.keras"
METRICS_SAVE_PATH = r"C:\Users\phili\epilepsy_mlops\metrics\model_metrics.json"

print(f"🧮 Starting model training process...")

#print(f"Data path: {BASE_DATA_PATH}")
#print(f"Model path: {BASE_MODEL_PATH}")

print(f"🧮 Starting model training process...")

# Create necessary directories
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)

# Load the data
print(f"📂 Loading training data from {PROCESSED_X_TRAIN_PATH}")
X_train = np.load(PROCESSED_X_TRAIN_PATH)
Y_train = np.load(PROCESSED_Y_TRAIN_PATH)
X_test = np.load(PROCESSED_X_TEST_PATH)
Y_test = np.load(PROCESSED_Y_TEST_PATH)

# Make sure data is in the right shape
print(f"📊 X_train shape: {X_train.shape}")

if len(X_train.shape) == 2:
    print("⚠️ Reshaping data for LSTM...")
    X_train = X_train.reshape(-1, 178, 1)
    X_test = X_test.reshape(-1, 178, 1)

# Define the model
print("🧠 Building LSTM model...")
model = Sequential([
    LSTM(56, input_shape=(178, 1), return_sequences=True),
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

model.summary()

# MLflow logging: Start a new run
with mlflow.start_run():

    # Log model parameters
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("input_shape", (178, 1))
    mlflow.log_param("n_epochs", 20)
    mlflow.log_param("batch_size", 15)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    print("Training model")
    hist = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=15,
        validation_data=(X_test, Y_test),
        shuffle=False,
        callbacks=[early_stopping]
    )

    # Log metrics to MLflow
    mlflow.log_metric("train_accuracy", float(hist.history["accuracy"][-1]))
    mlflow.log_metric("val_accuracy", float(hist.history["val_accuracy"][-1]))
    mlflow.log_metric("train_loss", float(hist.history["loss"][-1]))
    mlflow.log_metric("val_loss", float(hist.history["val_loss"][-1]))

    # Save the model
    print(f"💾 Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    mlflow.keras.log_model(model, "epilepsy_model")

    print(f"✅ Model saved in .keras format at {MODEL_SAVE_PATH}")

    # Save metrics
    metrics = {
        "train_accuracy": float(hist.history["accuracy"][-1]),
        "val_accuracy": float(hist.history["val_accuracy"][-1]),
        "train_loss": float(hist.history["loss"][-1]),
        "val_loss": float(hist.history["val_loss"][-1]),
        "epochs": len(hist.history["loss"]),
        "batch_size": 15,
        "architecture": "2xLSTM_+_Dense"
    }

    print(f"📈 Final metrics: Validation accuracy: {metrics['val_accuracy']:.4f}")

    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump({"model_path": MODEL_SAVE_PATH, "metrics": metrics}, f)

    print(f"✅ Training complete. Metrics saved in '{METRICS_SAVE_PATH}'")
