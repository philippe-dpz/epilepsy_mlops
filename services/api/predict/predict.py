import os
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_patient(patient_id, data_path, model_path):
    # Load data
    print(f"📂 Loading patient data from: {data_path}")
    df = pd.read_csv(data_path)

    # Check for valid patient
    df_patient = df[df["patient_id"] == patient_id]
    if df_patient.empty:
        return None, f"❌ No data found for patient {patient_id}."

    # Load model
    print(f"🧠 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Preprocess
    X = df_patient.drop(columns=["patient_id"]).values
    X = X.reshape(-1, X.shape[1], 1)

    # Predict
    preds = model.predict(X)
    pred_classes = np.argmax(preds, axis=1)

    # Get epileptic row indices (1 to 8)
    epileptic_idxs = [i + 1 for i, pred in enumerate(pred_classes) if pred == 1]

    # Format response
    message = (
        f"⚠️ Patient {patient_id} predicted epileptic recordings at rows: {epileptic_idxs}"
        if epileptic_idxs else
        f"✅ Patient {patient_id} predicted as non-epileptic in all recordings."
    )

    return {
        "patient_id": patient_id,
        "epileptic_recordings": epileptic_idxs,
        "message": message
    }, None
