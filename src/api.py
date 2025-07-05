# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import extract_cv

app = FastAPI()

class CVRequest(BaseModel):
    cv_text: str

@app.post("/extract")
def extract_entities(request: CVRequest):
    extracted = extract_cv(request.cv_text)
    return {"data": extracted}
