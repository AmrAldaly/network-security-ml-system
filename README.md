<div align="center">

# 🛡️ Network Security — Phishing URL Detection

### An end-to-end MLOps pipeline for real-time phishing URL classification

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Integrated-FF6B35?style=for-the-badge)](https://dagshub.com/AmrAldaly/network-security-ml-system)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 📌 Overview

This project implements a **production-grade MLOps pipeline** that detects phishing URLs in real time using machine learning. A URL is submitted to a REST API and classified as one of three outcomes:

| Prediction | Value | Meaning |
|:---:|:---:|---|
| **Legitimate** | `1` | Safe to visit |
| **Suspicious** | `0` | Exercise caution |
| **Phishing** | `-1` | Malicious — do not visit |

The system is built around clean separation of concerns: a modular **training pipeline** (ingestion → validation → transformation → training), a **feature extraction engine** that derives 30 URL signals in real time, and a **FastAPI inference layer** that serves predictions via REST. Experiment tracking is handled by **MLflow + DagsHub**, and the application is fully **Dockerized** and ready for cloud deployment.

> **Project Status:** Dockerized and ready for deployment. Not yet deployed to a live environment.

---

## 🏆 Model Performance

The best model selected through grid-search cross-validation was a **Random Forest Classifier**.

| Split | F1 Score | Precision | Recall |
|:---:|:---:|:---:|:---:|
| **Train** | `0.9911` | `0.9877` | `0.9945` |
| **Test** | `0.9749` | `0.9733` | `0.9765` |

The tight gap between train and test metrics confirms the model generalises well with no significant overfitting. Five classifiers were evaluated (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost) and the best was selected automatically based on test F1 score.

---

## 🚨 Problem Statement

Phishing attacks are among the most prevalent and damaging cybersecurity threats. Attackers craft deceptive URLs that closely mimic legitimate websites to harvest credentials, financial data, and personal information. Manual detection is neither scalable nor fast enough to keep up with the volume of new threats.

This project addresses the problem by engineering a system that:
- Extracts **30 structural and behavioural signals** from any URL automatically
- Classifies it in real time using a trained ensemble model
- Serves the prediction through a clean REST API, ready to be integrated into browsers, firewalls, or security toolchains

---

## 🏗️ System Architecture

### ETL Pipeline

Raw data originates from a local CSV file, is converted to JSON records, and loaded into **MongoDB Atlas** as the centralised data source for the training pipeline.

```
Local CSV Dataset
      │
      ▼  Convert rows → JSON documents
      ▼
  MongoDB Atlas  ←── Single source of truth for training data
```

---

### Training Pipeline

```
MongoDB Atlas
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Ingestion                                              │
│  · Pulls collection from MongoDB                             │
│  · Exports raw data to Feature Store                         │
│  · Splits into train.csv / test.csv                          │
└────────────────────────────┬─────────────────────────────────┘
                             │  DataIngestionArtifact
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Validation                                             │
│  · Validates column count and names against schema.yaml      │
│  · Detects distribution drift via KS test                    │
│  · Outputs drift report + valid / invalid file paths         │
└────────────────────────────┬─────────────────────────────────┘
                             │  DataValidationArtifact
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Transformation                                         │
│  · KNN Imputer — fills missing values                        │
│  · Saves preprocessor.pkl → final_model/                     │
│  · Outputs train.npy and test.npy                            │
└────────────────────────────┬─────────────────────────────────┘
                             │  DataTransformationArtifact
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Model Trainer                                               │
│  · Grid-searches 5 classifiers (RF, DT, GBM, LR, AdaBoost)  │
│  · Selects best model by test F1 score                       │
│  · Logs all metrics and model to MLflow / DagsHub            │
│  · Saves model.pkl → final_model/                            │
└────────────────────────────┬─────────────────────────────────┘
                             │  ModelTrainerArtifact
                             ▼
                   ✅ final_model/ ready for inference
```

---

### Inference Pipeline

```
Raw URL (string)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature Extraction  (src/utils/feature_extraction.py)       │
│  · 30 features extracted via regex, requests,                │
│    BeautifulSoup, python-whois, and SSL inspection           │
│  · All values encoded as {-1, 0, 1}                          │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Preprocessor  (final_model/preprocessor.pkl)                │
│  · KNN Imputer applied via .transform() only                 │
│  · Same object fitted during training — never refit          │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Model Inference  (final_model/model.pkl)                    │
│  · Random Forest Classifier                                  │
│  · Outputs prediction: -1 / 0 / 1                            │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
              { features, prediction, label }
```

---

### API Layer

```
POST /predict  ──▶  FastAPI (src/api/app.py)
                          │
                    Pydantic validates input
                          │
                    PredictionPipeline
                    ├── Feature Extraction
                    ├── preprocessor.transform()
                    └── model.predict()
                          │
                    ◀── JSON Response
```

---

### Deployment Architecture (Docker + AWS)

> The application is Dockerized and the architecture below is the intended deployment target.

```
Developer pushes to GitHub
          │
          ▼
  GitHub Actions CI/CD
  ├── Build Docker image
  └── Push to AWS ECR
          │
          ▼
      AWS EC2
  Pull image from ECR
  Run container → FastAPI on port 8080
```

---

## 📁 Project Structure

```
NetworkSecurityProject/
│
├── Artifacts/                          # Timestamped pipeline stage outputs
│   ├── DataIngestion/
│   ├── DataValidation/                 # drift_report.yaml, valid/invalid CSVs
│   └── DataTransformation/             # train.npy, test.npy
│
├── data_schema/
│   └── schema.yaml                     # Expected columns and data types
│
├── final_model/                        # Inference-ready artefacts
│   ├── model.pkl                       # Trained Random Forest classifier
│   └── preprocessor.pkl               # Fitted KNN Imputer pipeline
│
├── logs/                               # Daily rotating log files
├── mlruns/                             # MLflow local tracking data
│
├── Network_Data/
│   └── PhisingData.csv                 # Raw labelled dataset (30 features + Result)
│
├── src/
│   ├── api/
│   │   └── app.py                      # FastAPI app — serving layer only, no ML logic
│   │
│   ├── components/                     # One file per pipeline stage
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py        # Orchestrates all training stages
│   │   └── prediction_pipeline.py      # Orchestrates inference stages
│   │
│   ├── utils/
│   │   ├── feature_extraction.py       # 30-feature extractor for live URLs
│   │   ├── main_utils/
│   │   │   └── utils.py                # save_object, load_object, I/O helpers
│   │   └── ml_utils/
│   │       ├── metric/
│   │       │   └── classification_metric.py
│   │       └── model/
│   │           └── estimator.py        # NetworkModel wrapper
│   │
│   ├── cloud/                          # AWS S3 / Azure Blob (model pusher)
│   ├── constant/
│   │   └── training_pipeline/
│   │       └── __init__.py             # All pipeline constants and paths
│   ├── entity/
│   │   ├── artifact_entity.py          # Dataclasses for stage outputs
│   │   └── config_entity.py            # Dataclasses for stage configs
│   ├── exception/
│   │   └── exception.py                # Custom exception with full traceback
│   └── logging/
│       └── logger.py                   # Rotating daily file logger
│
├── .dockerignore
├── .env                                # Secrets — never commit this file
├── .gitignore
├── Dockerfile
├── main.py                             # Alternative training entry point
├── mongodb_connection.py
├── push_data.py                        # One-time CSV → MongoDB migration
├── requirements.txt
└── setup.py
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier is sufficient)
- Docker Desktop (for containerised runs)

### 1. Clone the Repository

```bash
git clone https://github.com/AmrAldaly/network-security-ml-system.git
cd network-security-ml-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root — this file is listed in `.gitignore` and must never be committed:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/
```

### 5. Load Data into MongoDB (first time only)

```bash
python push_data.py
```

### 6. Run the Training Pipeline

```bash
python src/pipeline/training_pipeline.py
```

This executes all stages in sequence: Data Ingestion → Validation → Transformation → Model Training. On completion, `final_model/model.pkl` and `final_model/preprocessor.pkl` will be created.

### 7. Start the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```

The interactive API docs will be available at **http://localhost:8080/docs**

---

## 🐳 Docker

### Build

```bash
docker build -t phishing-detector .
```

### Run

```bash
docker run -p 8080:8080 --env-file .env phishing-detector
```

The API will be available at **http://localhost:8080**

---

## 🔌 API Reference

### `POST /predict`

Classifies a URL and returns the extracted features alongside the prediction.

**Request**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://paypal-secure-login.com/verify"}'
```

**Response**

```json
{
  "features": {
    "having_IP_Address": 1,
    "URL_Length": -1,
    "Shortining_Service": 1,
    "having_At_Symbol": 1,
    "double_slash_redirecting": 1,
    "Prefix_Suffix": -1,
    "having_Sub_Domain": -1,
    "SSLfinal_State": 0,
    "Domain_registeration_length": -1,
    "Favicon": -1,
    "port": 1,
    "HTTPS_token": -1,
    "Request_URL": -1,
    "URL_of_Anchor": -1,
    "Links_in_tags": -1,
    "SFH": -1,
    "Submitting_to_email": 1,
    "Abnormal_URL": -1,
    "Redirect": 1,
    "on_mouseover": 1,
    "RightClick": -1,
    "popUpWidnow": 1,
    "Iframe": 1,
    "age_of_domain": -1,
    "DNSRecord": 1,
    "web_traffic": 0,
    "Page_Rank": -1,
    "Google_Index": -1,
    "Links_pointing_to_page": -1,
    "Statistical_report": -1
  },
  "prediction": -1,
  "label": "phishing"
}
```

### `GET /`

Health check endpoint.

```bash
curl http://localhost:8080/
# → {"status": "ok", "message": "Phishing Detection API is running."}
```

---

## 📊 Feature Engineering

The model is built on **30 features** extracted directly from the URL and its associated web page. Every feature is encoded as `-1` (phishing signal), `0` (suspicious/unknown), or `1` (legitimate signal).

| # | Feature | Encoding Logic |
|:---:|---|---|
| 1 | `having_IP_Address` | IP in host → `-1`, domain name → `1` |
| 2 | `URL_Length` | `<54` → `1`, `54–75` → `0`, `>75` → `-1` |
| 3 | `Shortining_Service` | Shortener detected → `-1` |
| 4 | `having_At_Symbol` | `@` in URL → `-1` |
| 5 | `double_slash_redirecting` | `//` after position 6 → `-1` |
| 6 | `Prefix_Suffix` | `-` in domain → `-1` |
| 7 | `having_Sub_Domain` | 1 dot → `1`, 2 dots → `0`, 3+ → `-1` |
| 8 | `SSLfinal_State` | Valid HTTPS cert → `1`, missing → `-1` |
| 9 | `Domain_registeration_length` | Registered ≤1 year → `-1` |
| 10 | `Favicon` | Favicon from external domain → `-1` |
| 11 | `port` | Non-standard port → `-1` |
| 12 | `HTTPS_token` | `https` token in domain name → `-1` |
| 13 | `Request_URL` | `>61%` external resources → `-1` |
| 14 | `URL_of_Anchor` | `>67%` external/null anchors → `-1` |
| 15 | `Links_in_tags` | `>17%` external meta/script/link → `-1` |
| 16 | `SFH` | Form action blank or external → `-1` |
| 17 | `Submitting_to_email` | Form uses `mailto:` → `-1` |
| 18 | `Abnormal_URL` | Host ≠ whois registered domain → `-1` |
| 19 | `Redirect` | `≤1` redirect → `1`, `2–4` → `0`, `>4` → `-1` |
| 20 | `on_mouseover` | Status bar URL spoofed → `-1` |
| 21 | `RightClick` | Right-click disabled → `-1` |
| 22 | `popUpWidnow` | Credential-harvesting popup → `-1` |
| 23 | `Iframe` | Hidden iframe present → `-1` |
| 24 | `age_of_domain` | Domain `<6` months old → `-1` |
| 25 | `DNSRecord` | No DNS record found → `-1` |
| 26 | `web_traffic` | Unknown (Alexa deprecated) → `0` |
| 27 | `Page_Rank` | Unknown (Google API deprecated) → `-1` |
| 28 | `Google_Index` | Not indexed by Google → `-1` |
| 29 | `Links_pointing_to_page` | `0` links → `-1`, `1–2` → `0`, `2+` → `1` |
| 30 | `Statistical_report` | In phishing DB → `-1` (default conservative) |

---

## 🧪 Experiment Tracking

All training runs are logged automatically to **DagsHub** via MLflow:

- **Metrics tracked:** `train_f1_score`, `train_precision`, `train_recall`, `test_f1_score`, `test_precision`, `test_recall`
- **Model registration:** each model is registered under its algorithm name (e.g. `Random Forest`) in the DagsHub Model Registry
- **Reproducibility:** every run is timestamped and linked to its hyperparameters and dataset version

🔗 **Experiment Dashboard:** [dagshub.com/AmrAldaly/network-security-ml-system](https://dagshub.com/AmrAldaly/network-security-ml-system)

---

## 🔮 Roadmap

| Priority | Improvement |
|:---:|---|
| High | Integrate PhishTank / OpenPhish API to make `Statistical_report` dynamic |
| High | Add AWS deployment workflow (GitHub Actions → ECR → EC2) |
| Medium | Async feature extraction with `asyncio` to reduce inference latency |
| Medium | Batch prediction endpoint — `POST /predict/batch` for lists of URLs |
| Medium | Automated retraining trigger when data drift exceeds a configured threshold |
| Low | API key authentication on the FastAPI layer |
| Low | Lightweight frontend dashboard for manual URL submission |

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Built by <a href="https://github.com/AmrAldaly">Amr Aldaly</a>
</div>
