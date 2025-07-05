from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from predict.predict import predict_patient

app = FastAPI()

# Paths
BASE_DATA_PATH = "C:/Users/phili/epilepsy_mlops/data"
PATIENT_DATA_PATH = os.path.join(BASE_DATA_PATH, "patients/patients_data_updated.csv")
MODEL_PATH = "C:/Users/phili/epilepsy_mlops/models/epilepsy_model.keras"


@app.get("/predict/{patient_id}")
def predict(patient_id: int):
    result, error = predict_patient(patient_id, PATIENT_DATA_PATH, MODEL_PATH)
    if error:
        return JSONResponse(status_code=404, content={"error": error})
    return result
