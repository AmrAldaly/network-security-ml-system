# 🛡️ Network Security — Phishing URL Detection (MLOps)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)
![DagsHub](https://img.shields.io/badge/DagsHub-integrated-purple)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![AWS](https://img.shields.io/badge/AWS-ECR%20%2B%20EC2-orange)

---

## 📌 Project Overview

This project is a **production-grade MLOps pipeline** for detecting **phishing URLs** in real time. It classifies any given URL into one of three categories:

| Label | Value | Meaning |
|---|---|---|
| `legitimate` | `1` | Safe URL |
| `suspicious` | `0` | Borderline — exercise caution |
| `phishing` | `-1` | Malicious URL — do not visit |

The system is built with a clean separation between training, inference, and serving, and is deployed to **AWS EC2** via a **GitHub Actions CI/CD pipeline** with Docker images pushed to **AWS ECR**.

---

## 🚨 Problem Statement

Phishing attacks remain one of the most prevalent cybersecurity threats. Attackers craft URLs that mimic legitimate websites to steal credentials, financial data, and personal information. Manual detection does not scale — this project automates URL classification using **30 URL-based and HTML-based features** and an ensemble ML model trained on a labelled dataset of phishing and legitimate URLs.

---

## 🏗️ Architecture

### ETL Pipeline (Data Loading)

```
Source (Local CSV / APIs / S3 / Internal DB)
         │
         ▼  Basic Preprocessing & Cleaning
         │  Convert rows → JSON records
         ▼
     MongoDB Atlas  ←──── Raw data stored as JSON documents
```

### Training Pipeline

```
MongoDB Atlas (PhisingData collection)
      │
      ▼
┌─────────────────────────┐
│   Data Ingestion        │  ← Pulls from MongoDB, exports to Feature Store (raw.csv)
│                         │    Drops unused columns, splits → train.csv / test.csv
└────────────┬────────────┘
             │  DataIngestionArtifact
             ▼
┌─────────────────────────┐
│   Data Validation       │  ① Validates schema (same no. of features)
│                         │  ② Detects data drift (distribution shift)
│                         │  ③ Checks numerical column existence
│                         │    Outputs: valid/invalid CSVs + drift report
└────────────┬────────────┘
             │  DataValidationArtifact
             ▼
┌─────────────────────────┐
│   Data Transformation   │  ① KNN Imputer   — fills missing values
│                         │  ② Robust Scaler — scales using median/IQR
│                         │  ③ SMOTE-Tomek   — resamples for class balance (train only)
│                         │    Saves: preprocessor.pkl, train.npy, test.npy
└────────────┬────────────┘
             │  DataTransformationArtifact
             ▼
┌─────────────────────────┐
│   Model Trainer         │  ← Grid-search across 5 classifiers (RF, DT, GBM, LR, AdaBoost)
│                         │    Selects best model by F1 score
│                         │    Logs metrics + model to MLflow / DagsHub
│                         │    Saves: model.pkl + preprocessor.pkl → final_model/
└────────────┬────────────┘
             │  ModelTrainerArtifact
             ▼
┌─────────────────────────┐
│   Model Evaluation      │  ← Compares new model score vs expected accuracy threshold
│                         │    Accepts or rejects the new model
└────────────┬────────────┘
             │  Model Accepted? Yes / No
             ▼
┌─────────────────────────┐
│   Model Pusher          │  ← If accepted: pushes artefacts to Cloud (AWS S3 / Azure)
│                         │    If rejected: pipeline halts, existing model stays live
└─────────────────────────┘
```

### Inference Pipeline

```
URL (string)
     │
     ▼
┌──────────────────────────────┐
│   Feature Extraction         │  ← 30 features via regex, requests,
│   src/utils/feature_         │    BeautifulSoup, whois, SSL checks
│   extraction.py              │    Output: {-1, 0, 1} encoded dict
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   KNN Imputer (no-op)        │  ← Part of fitted preprocessor.pkl
│   + Robust Scaler            │    .transform() only — never .fit_transform()
│   preprocessor.pkl           │    Same statistics fitted during training
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Model Inference            │  ← Trained classifier → outputs -1 / 0 / 1
│   model.pkl                  │    SMOTE-Tomek is NOT applied at inference
└──────────────┬───────────────┘
               │
               ▼
         JSON Response
   { features, prediction, label }
```

> ⚠️ **Preprocessing alignment:** Training applies `KNNImputer → RobustScaler → SMOTETomek`.
> Inference applies only `KNNImputer → RobustScaler` (same fitted object, `.transform()` only).
> SMOTE-Tomek is a training-only resampler and is **never** used at inference time.

### API Layer

```
Client  ──POST /predict──▶  FastAPI (src/api/app.py)
                                    │  Pydantic validation
                             PredictionPipeline
                                    │
                     ┌──────────────┼──────────────┐
                     │              │               │
              Feature           Preprocessor     Model
              Extraction         .transform()   .predict()
                     └──────────────┼──────────────┘
                                    │
                            ◀── JSON Response
```

### Deployment (CI/CD → AWS)

```
GitHub Push
     │
     ▼
GitHub Actions
  ├── CI: Lint & Test
  └── CD: Build Docker Image
               │
               ▼
          AWS ECR  ← Docker image pushed
               │
               ▼
          AWS EC2  ← Container pulled and run  (or AWS App Runner)
               │
               ▼
     FastAPI serving on port 8080
```

---

## 📁 Folder Structure

```
NetworkSecurityProject/
│
├── Artifacts/                      # Timestamped outputs from each pipeline stage
│   ├── DataIngestion/
│   ├── DataValidation/             # Includes drift report
│   └── DataTransformation/         # train.npy, test.npy, preprocessor.pkl
├── data_schema/
│   └── schema.yaml                 # Column names, types, expected value ranges
├── final_model/
│   ├── model.pkl                   # Best trained classifier (raw estimator)
│   └── preprocessor.pkl            # Fitted KNNImputer + RobustScaler pipeline
├── logs/                           # Daily rotating log files
├── mlruns/                         # MLflow local experiment data
├── Network_Data/
│   └── PhisingData.csv             # Raw labelled dataset (30 features + Result)
│
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI serving layer — NO ML logic here
│   │
│   ├── components/
│   │   ├── data_ingestion.py       # MongoDB → Feature Store → train/test CSV
│   │   ├── data_validation.py      # Schema check, drift detection
│   │   ├── data_transformation.py  # KNNImputer + RobustScaler + SMOTETomek
│   │   └── model_trainer.py        # Grid-search, MLflow logging, saves final_model/
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Chains all training components end-to-end
│   │   └── prediction_pipeline.py  # Inference: feature → preprocess → predict
│   │
│   ├── utils/
│   │   ├── feature_extraction.py   # Extracts all 30 features from a live URL
│   │   ├── main_utils/utils.py     # save_object, load_object, load_numpy_array
│   │   └── ml_utils/
│   │       ├── metric/classification_metric.py  # F1, Precision, Recall
│   │       └── model/estimator.py               # NetworkModel wrapper
│   │
│   ├── cloud/                      # AWS S3 / Azure Blob integration (Model Pusher)
│   ├── constant/training_pipeline/ # Pipeline-wide constants (__init__.py)
│   ├── entity/
│   │   ├── artifact_entity.py      # Dataclasses for component outputs
│   │   └── config_entity.py        # Dataclasses for component configs
│   ├── exception/exception.py      # Custom exception with traceback info
│   └── logging/logger.py           # Rotating daily log configuration
│
├── .dockerignore
├── .env                            # MONGODB_URL and secrets (never commit)
├── .gitignore
├── Dockerfile                      # Builds FastAPI image, uvicorn on port 8080
├── main.py                         # Entry point: runs full training pipeline
├── mongodb_connection.py           # MongoDB Atlas connection helper
├── push_data.py                    # Migrates local CSV → MongoDB
├── README.md
├── requirements.txt
└── setup.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AmrAldaly/network-security-ml-system.git
cd network-security-ml-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_REGION=us-east-1
```

### 5. Push data to MongoDB (first time only)

```bash
python push_data.py
```

### 6. Run the training pipeline

```bash
python main.py
```

Executes: Ingestion → Validation → Transformation → Training → Evaluation → Push.
Saves `final_model/model.pkl` and `final_model/preprocessor.pkl`.

### 7. Start the API server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```

Interactive docs: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🐳 Docker

### Build the image

```bash
docker build -t phishing-detector .
```

### Run the container

```bash
docker run -p 8080:8080 --env-file .env phishing-detector
```

---

## ☁️ AWS Deployment (CI/CD)

The project uses **GitHub Actions** for automated deployment to AWS:

1. **CI** — runs tests and linting on every push
2. **CD** — builds Docker image → pushes to **AWS ECR**
3. **AWS EC2** — pulls latest image from ECR → restarts container

To configure, add these secrets to your GitHub repository:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY_URI` | Full ECR repository URI |
| `EC2_HOST` | EC2 public IP or DNS |
| `EC2_USERNAME` | e.g. `ec2-user` or `ubuntu` |
| `EC2_SSH_KEY` | PEM key contents (base64) |

Push to `main` → workflow triggers automatically.

---

## 🔌 API Usage

### Endpoint

```
POST /predict
```

### Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.paypal-secure-login.com/verify"}'
```

### Response

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

### Health check

```bash
curl http://localhost:8080/
# {"status": "ok", "message": "Phishing Detection API is running."}
```

---

## 📊 Features

All 30 features are encoded as `-1` (phishing), `0` (suspicious), or `1` (legitimate):

| # | Feature | Description |
|---|---|---|
| 1 | `having_IP_Address` | URL uses raw IP instead of a domain name |
| 2 | `URL_Length` | < 54 chars (1), 54–75 (0), > 75 (-1) |
| 3 | `Shortining_Service` | URL shortener service detected |
| 4 | `having_At_Symbol` | `@` present in URL |
| 5 | `double_slash_redirecting` | `//` occurs after position 6 |
| 6 | `Prefix_Suffix` | `-` in the domain name |
| 7 | `having_Sub_Domain` | 1 dot (1), 2 dots (0), 3+ dots (-1) |
| 8 | `SSLfinal_State` | HTTPS with a valid certificate |
| 9 | `Domain_registeration_length` | Domain registered for ≤ 1 year |
| 10 | `Favicon` | Favicon loaded from an external domain |
| 11 | `port` | Non-standard port in URL |
| 12 | `HTTPS_token` | `https` token appears inside the domain name |
| 13 | `Request_URL` | % of page resources loaded from external domains |
| 14 | `URL_of_Anchor` | % of anchors pointing to external or null targets |
| 15 | `Links_in_tags` | % of meta/script/link tags referencing external URLs |
| 16 | `SFH` | Form action is blank, `about:blank`, or external |
| 17 | `Submitting_to_email` | Form uses a `mailto:` action |
| 18 | `Abnormal_URL` | URL host doesn't match whois-registered domain |
| 19 | `Redirect` | ≤1 redirect (1), 2–4 (0), > 4 (-1) |
| 20 | `on_mouseover` | Status bar URL hidden via `onMouseOver` |
| 21 | `RightClick` | Right-click disabled on the page |
| 22 | `popUpWidnow` | Popup window requesting user credentials |
| 23 | `Iframe` | Hidden iframe (zero width/height) present |
| 24 | `age_of_domain` | Domain younger than 6 months |
| 25 | `DNSRecord` | No DNS record found for domain |
| 26 | `web_traffic` | Traffic rank — 0 (unknown, Alexa API deprecated) |
| 27 | `Page_Rank` | Google PageRank — -1 (unknown, API deprecated) |
| 28 | `Google_Index` | URL is indexed by Google |
| 29 | `Links_pointing_to_page` | Count of inbound links on the page |
| 30 | `Statistical_report` | Listed in phishing statistical report databases |

---

## 🧪 MLflow / DagsHub Tracking

Training runs are logged to **DagsHub** automatically:

- **Metrics:** `train_f1_score`, `test_f1_score`, `test_precision`, `test_recall`
- **Model name:** registered under its actual algorithm name (e.g. `Random Forest`)
- **Artefacts:** model uploaded to DagsHub Model Registry

View experiments: `https://dagshub.com/AmrAldaly/network-security-ml-system`

---

## 🔮 Future Improvements

- **Real-time threat feeds** — integrate PhishTank / OpenPhish API for `Statistical_report`
- **Traffic & PageRank APIs** — replace defaulted features with SimilarWeb / Moz API calls
- **Async feature extraction** — parallelise HTTP/whois calls with `asyncio` for lower latency
- **Batch prediction endpoint** — `POST /predict/batch` accepting a list of URLs
- **Model retraining trigger** — auto-retrain when drift report exceeds a configured threshold
- **Authentication** — add API key or OAuth2 to the FastAPI layer
- **Frontend dashboard** — simple React UI for manual URL checking
