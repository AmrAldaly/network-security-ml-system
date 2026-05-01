"""
prediction_pipeline.py
-----------------------
Orchestrates the full inference flow:
    URL → feature extraction → preprocessing → model prediction → label

This module is the ONLY place that ties together the feature extractor,
the preprocessor, and the trained model. The API layer (app.py) calls this
and never contains any ML logic directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING vs INFERENCE PREPROCESSING — ALIGNMENT NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

During training (data_transformation.py), the preprocessing pipeline is:
    1. KNN Imputer — fills NaN values using k-nearest neighbour strategy

The saved preprocessor.pkl = Pipeline([("imputer", KNNImputer)]) only.
No RobustScaler is applied — features are already encoded as {-1, 0, 1}
so scaling would distort the discrete encoding.

During inference:
    - Feature extractor always outputs clean {-1, 0, 1} with no NaN values,
      so KNNImputer is effectively a no-op, but data must still pass through
      it to maintain pipeline compatibility.

Target label encoding (same at training AND inference):
    -1 → phishing
     0 → suspicious
     1 → legitimate

⚠️  PREREQUISITE: final_model/preprocessor.pkl is saved by
    data_transformation.py. Run the full training pipeline before
    starting the API, or this will raise FileNotFoundError at startup.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import numpy as np
from pathlib import Path

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import load_object
from src.utils.feature_extraction import extract_features, FEATURE_ORDER

# ── paths to saved artefacts (written by training pipeline) ──────────────────
# Paths are anchored to the project root using Path(__file__) so they resolve
# correctly regardless of which directory Python is launched from.
# This file lives at: src/pipeline/prediction_pipeline.py
# Project root is 3 parents up.
_PROJECT_ROOT     = Path(__file__).resolve().parents[2]
MODEL_PATH        = _PROJECT_ROOT / "final_model" / "model.pkl"
PREPROCESSOR_PATH = _PROJECT_ROOT / "final_model" / "preprocessor.pkl"

# ── label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    -1: "phishing",
     0: "suspicious",
     1: "legitimate",
}


class PredictionPipeline:
    """
    Loads the trained model and preprocessor once at construction time,
    then exposes a `predict` method for repeated inference.

    Preprocessing alignment:
        Training  → KNNImputer (fit_transform on train features)
        Inference → KNNImputer (transform only — same fitted object, never refit)

    Target label encoding (preserved from training):
        -1 → phishing | 0 → suspicious | 1 → legitimate
    """

    def __init__(self):
        try:
            logging.info("Loading model and preprocessor from disk...")
            self.model        = load_object(MODEL_PATH)
            self.preprocessor = load_object(PREPROCESSOR_PATH)
            logging.info("Model and preprocessor loaded successfully.")
        except FileNotFoundError as e:
            raise CustomException(
                f"Artefact not found: {e}.\n"
                "Ensure model_trainer.py saves preprocessor.pkl to final_model/ "
                "using: save_object('final_model/preprocessor.pkl', preprocessor)",
                sys,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, url: str) -> dict:
        """
        End-to-end prediction for a single URL.

        Parameters
        ----------
        url : str
            The URL to classify (scheme optional — will be added if missing).

        Returns
        -------
        dict with keys:
            features   : dict  — all 30 extracted feature values (-1/0/1)
            prediction : int   — raw model output (-1, 0, or 1)
            label      : str   — human-readable label

        Preprocessing steps applied (mirrors training exactly):
            1. feature_extraction.py  → {-1, 0, 1} encoded feature dict
            2. KNNImputer (no-op)     → features are complete; passes through pipeline
            3. model.predict()        → classifier outputs -1 / 0 / 1
        """
        try:
            # ── Step 1: Feature Extraction ────────────────────────────────
            # Produces a dict of {feature_name: int} in FEATURE_ORDER.
            # All values are {-1, 0, 1} — no NaN will be present.
            logging.info(f"Extracting features for URL: {url}")
            features: dict = extract_features(url)

            # ── Step 2: Build numpy array in the exact training column order ──
            # Column order MUST match what data_transformation.py used when
            # fitting the preprocessor. FEATURE_ORDER in feature_extraction.py
            # is the canonical source of truth for this ordering.
            feature_array = np.array(
                [[features[col] for col in FEATURE_ORDER]],
                dtype=float,
            )

            # ── Step 3: Apply fitted preprocessor (KNNImputer only) ────────────
            # We call .transform() only — NEVER .fit_transform().
            # Re-fitting would use inference-time statistics and break alignment
            # with the model's training distribution.
            logging.info("Applying preprocessor (KNNImputer)...")
            transformed = self.preprocessor.transform(feature_array)

            # ── Step 4: Model Inference ───────────────────────────────────────
            # The model was trained on KNN-imputed {-1, 0, 1} features.
            # We feed it the same imputed feature array.
            logging.info("Running model inference...")
            prediction = int(self.model.predict(transformed)[0])

            label = LABEL_MAP.get(prediction, "unknown")
            logging.info(f"Prediction result: {prediction} ({label})")

            return {
                "features":   features,
                "prediction": prediction,
                "label":      label,
            }

        except Exception as e:
            raise CustomException(e, sys)
