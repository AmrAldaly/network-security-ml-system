"""
app.py
------
FastAPI serving layer for the phishing URL detection model.

Responsibilities:
  - Accept HTTP POST requests at /predict
  - Validate input with Pydantic
  - Delegate ALL prediction logic to PredictionPipeline
  - Return a clean JSON response

This file contains NO machine learning logic.
"""

import os
import sys
from pathlib import Path

# ── Working directory fix ─────────────────────────────────────────────────────
# When app.py is run directly (python src/api/app.py), CWD is src/api/.
# All relative paths in the codebase (final_model/, data_schema/, Artifacts/)
# are relative to the PROJECT ROOT, not this file's directory.
#
# We anchor CWD to the project root here — before any src.* imports — so that
# every module that uses relative paths works correctly.
# This file lives at: src/api/app.py → project root is 2 parents up.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_PROJECT_ROOT)
# Also ensure the project root is on sys.path so src.* imports resolve.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Dict

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logging.logger import logging

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Classifies a URL as **phishing** (-1), **suspicious** (0), or **legitimate** (1) "
        "using a trained ML model with 30 URL-based features."
    ),
    version="1.0.0",
)

# Instantiate pipeline once at startup (model & preprocessor loaded into memory)
pipeline = PredictionPipeline()


# ── Request / Response schemas ────────────────────────────────────────────────

class URLRequest(BaseModel):
    """Input schema — accepts a raw URL string."""
    url: str

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL must not be empty.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{"url": "https://www.paypal-secure-login.com/verify"}]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema — features extracted, raw prediction, and human label."""
    features:   Dict[str, int]
    prediction: int
    label:      str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "features": {"having_IP_Address": 1, "URL_Length": -1, "...": 0},
                "prediction": -1,
                "label": "phishing",
            }]
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Phishing Detection API is running."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: URLRequest):
    """
    Classify a URL as phishing, suspicious, or legitimate.

    **Request body:**
    ```json
    { "url": "https://example.com" }
    ```

    **Response:**
    ```json
    {
      "features":   { "having_IP_Address": 1, ... },
      "prediction": 1,
      "label":      "legitimate"
    }
    ```
    """
    try:
        logging.info(f"Received prediction request for: {request.url}")
        result = pipeline.predict(request.url)
        return PredictionResponse(**result)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
