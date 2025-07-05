# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import extract_cv

app = FastAPI()

class CVRequest(BaseModel):
    cv_text: str

@app.get("/")
def read_root():
    return {"status": "CV extraction service is running"}

@app.post("/extract")
def extract_entities(request: CVRequest):
    try:
        extracted_data = extract_cv(request.cv_text)
        return {"data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
