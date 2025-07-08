from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from jose import JWTError, jwt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Simple Bearer scheme (no form)
bearer_scheme = HTTPBearer()

# Must match your auth_api's secret & algorithm
SECRET_KEY = "your_secret_key"  # Make sure this matches auth API
ALGORITHM = "HS256"

def verify_token(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise JWTError("No username in token")
        return username
    except JWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Load paths from env or use defaults
BASE_DATA_PATH = os.getenv("DATA_PATH", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./production/model/data/model.keras")
PATIENT_DATA_PATH = os.path.join(BASE_DATA_PATH, "patients_inference/patients_data_updated.csv")

# Global variables for model and data
df = None
model = None

@app.on_event("startup")
async def startup_event():
    global df, model
    try:
        # Load data
        if not os.path.exists(PATIENT_DATA_PATH):
            logger.error(f"Patient data file not found: {PATIENT_DATA_PATH}")
            raise FileNotFoundError(f"Patient data file not found: {PATIENT_DATA_PATH}")
        
        df = pd.read_csv(PATIENT_DATA_PATH)
        logger.info(f"Loaded patient data with shape: {df.shape}")
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise e

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "data_shape": df.shape if df is not None else None
    }

@app.get("/predict/{patient_id}")
def predict(patient_id: int, username: str = Depends(verify_token)):
    global df, model
    
    if df is None or model is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded")
    
    # Filter data for the specific patient
    df_patient = df[df["patient_id"] == patient_id]
    if df_patient.empty:
        raise HTTPException(status_code=404, detail=f"No data found for patient {patient_id}")
    
    try:
        # Prepare input data (remove patient_id column)
        X = df_patient.drop(columns=["patient_id"]).values
        logger.info(f"Input shape before reshape: {X.shape}")
        
        # Reshape for model input: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        logger.info(f"Input shape after reshape: {X.shape}")
        
        # Make predictions
        preds = model.predict(X)
        pred_classes = np.argmax(preds, axis=1)
        
        # Get epileptic recording indices (1-based)
        epileptic_idxs = [i+1 for i, c in enumerate(pred_classes) if c == 1]
        
        # Create response message
        message = (
            f"⚠️ Patient {patient_id} predicted epileptic recordings at rows: {epileptic_idxs}"
            if epileptic_idxs else
            f"✅ Patient {patient_id} predicted as non-epileptic in all recordings."
        )
        
        logger.info(f"Prediction completed for patient {patient_id} by user {username}")
        
        return {
            "patient_id": patient_id,
            "total_recordings": len(pred_classes),
            "epileptic_recordings": epileptic_idxs,
            "predictions": pred_classes.tolist(),
            "confidence_scores": preds.max(axis=1).tolist(),
            "message": message,
            "processed_by": username
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)