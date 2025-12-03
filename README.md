# Real Estate ML API

## 1. Overview

The Real Estate ML API is a lightweight FastAPI-based backend service that predicts residential property prices based on numerical and categorical input features (e.g. square footage, number of bedrooms, location).

It will serve a REST API for predictions, optionally retraining the model from new data, and provide an endpoint for model health and versioning.

This project demonstrates a real-world workflow combining:
• Machine learning model development (with scikit-learn)
• API design (with FastAPI)
• Model serving and persistence
• Deployment readiness (containerization, testing, environment configuration)

## 2. Goals and Objectives

### Primary Goals

• Build a predictive ML API that returns estimated house prices in JSON.
• Expose a /predict endpoint for users to send property feature data.
• Package model training and serving code into a clean, modular structure.
• Demonstrate reproducibility using uv + pyproject.toml.

### Secondary Goals

• Include /train route to retrain from CSV data uploads.
• Implement logging for model inputs and predictions.
• Support versioning and health endpoints (/health, /version).
• Prepare for optional containerization (Docker) and CI/CD pipeline.

## 3. Key Features

| Feature           | Description                                  | Status      |
| ----------------- | -------------------------------------------- | ----------- |
| /predict          | Accept JSON input and return predicted price | ✅ Core     |
| /train            | Accept new CSV data to retrain model         | 🔜 Optional |
| /health           | Health check endpoint for uptime monitoring  | ✅ Core     |
| /version          | Return model version and metadata            | ✅ Core     |
| Model persistence | Store trained model with joblib              | ✅ Core     |
| Validation        | Enforce schema with pydantic models          | ✅ Core     |
| Logging           | Log predictions and retraining events        | 🔜 Optional |

## 4. User Stories

    1.	As a data consumer, I want to send property details and receive a price prediction instantly.
    2.	As a developer, I want a modular and documented API that I can deploy to a cloud provider.
    3.	As a data scientist, I want to retrain the model easily on new data without code changes.
    4.	As a maintainer, I want simple monitoring endpoints to verify service health.

## 5. Technical Requirements

### Core Stack

| Layer             | Technology                          |
| ----------------- | ----------------------------------- |
| Backend Framework | FastAPI                             |
| Server            | Uvicorn                             |
| ML / Data         | scikit-learn, pandas, numpy, joblib |
| Environment       | uv (project + dependency manager)   |
| Validation        | pydantic                            |
| Configuration     | python-dotenv                       |
| Dev Tools         | pytest, ruff, httpx                 |

### Endpoints

| Method | Endpoint | Description                                                |
| ------ | -------- | ---------------------------------------------------------- |
| POST   | /predict | Send JSON property features and return predicted price     |
| POST   | /train   | (Optional) Upload new CSV dataset to retrain model         |
| GET    | /health  | Return status { "status": "ok" }                           |
| GET    | /version | Return current model metadata { "model_version": "1.0.0" } |

### Example /predict Request

```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1,
  "zipcode": 98052
}
```

### Example Response

```json
{
  "predicted_price": 525000,
  "model_version": "1.0.0",
  "timestamp": "2025-11-04T15:23:00Z"
}
```

## 6. Architecture Overview

```
real-estate-ml-api/
├── app/
│   ├── main.py          # FastAPI app entrypoint
│   ├── routes/
│   │   ├── predict.py   # Prediction endpoint
│   │   └── train.py     # Retrain endpoint (optional)
│   ├── models/
│   │   ├── regressor.pkl  # Trained model file
│   ├── schemas/
│   │   └── house.py     # Pydantic request/response models
│   └── utils/
│       └── preprocess.py # Data preparation utilities
├── data/
│   ├── housing.csv
├── model/
│   └── train_model.py   # Script to train and save model
├── tests/
│   ├── test_api.py
├── pyproject.toml
└── README.md
```

## 7. Deployment Plan

| Stage   | Task                                                | Tool               |
| ------- | --------------------------------------------------- | ------------------ |
| Dev     | Local run with uv run uvicorn app.main:app --reload | uv                 |
| Build   | Package Docker image                                | Docker             |
| Deploy  | Deploy to Render / Fly.io / Cloudflare Workers      | CI/CD              |
| Monitor | Log requests and add /health check                  | FastAPI middleware |

## 8. Success Metrics

| Metric                 | Target                            |
| ---------------------- | --------------------------------- |
| ✅ Prediction latency  | < 250 ms per request              |
| ✅ Model accuracy (R²) | ≥ 0.85 on test data               |
| ✅ Test coverage       | ≥ 80% for core routes             |
| ✅ Uptime              | 99%                               |
| ✅ API documentation   | Auto-generated by FastAPI Swagger |

## 9. Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/andrewgilliland/real-estate-ml-api.git
   cd real-estate-ml-api
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

### Training the Model

Before running the API, you need to train the machine learning model:

```bash
# Train the model using the housing data in data/housing.csv
uv run python model/train_model.py
```

This will:

- Load training data from `data/housing.csv`
- Train a RandomForestRegressor model
- Evaluate model performance (R², MAE, RMSE)
- Save the trained model to `app/models/regressor.pkl`
- Save model metadata to `app/models/metadata.json`

**Expected output:**

```
Loading training data...
Loaded 20 records.
Model R² on test set: 0.8934
✅ Model meets accuracy target!
Training R²: 0.9512
Testing R²: 0.8934
✅ Model saved to: app/models/regressor.pkl
✅ Metadata saved to: app/models/metadata.json
🎉 Model training completed successfully!
```

### Running the API Locally

Once the model is trained, start the FastAPI development server:

```bash
# Start the server with auto-reload
uv run uvicorn app.main:app --reload
```

The API will be available at:

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Testing the API

#### Option 1: Interactive API Documentation (Recommended)

Visit http://localhost:8000/docs in your browser to see the auto-generated API documentation. You can test all endpoints directly from the browser.

#### Option 2: cURL Commands

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Model Version:**

```bash
curl http://localhost:8000/version
```

**Make a Prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1,
    "zipcode": 98052
  }'
```

**Expected response:**

```json
{
  "predicted_price": 525000.0,
  "model_version": "1.0.0",
  "timestamp": "2025-12-02T14:30:45.123456"
}
```

#### Option 3: Python httpx

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict",
    json={
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1,
        "zipcode": 98052
    }
)

print(response.json())
```

### Project Structure

```
real-estate-ml-api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Trained model files
│   │   ├── regressor.pkl    # Trained RandomForest model
│   │   └── metadata.json    # Model metadata and performance
│   └── schemas/
│       └── house.py         # Pydantic request/response models
├── data/
│   └── housing.csv          # Training dataset
├── model/
│   └── train_model.py       # Model training script
├── tests/                   # Unit tests
├── pyproject.toml           # Project dependencies
├── uv.lock                  # Locked dependency versions
└── README.md
```

### Development Workflow

1. **Modify training data**: Edit `data/housing.csv`
2. **Retrain model**: `uv run python model/train_model.py`
3. **Restart API**: Server auto-reloads if using `--reload` flag
4. **Test changes**: Visit http://localhost:8000/docs

### Common Issues

**Issue: "Model not loaded" error when starting API**

- **Solution**: Train the model first with `uv run python model/train_model.py`

**Issue: Import errors**

- **Solution**: Ensure dependencies are installed with `uv sync`

**Issue: Model R² score below target (< 0.85)**

- **Solution**: Add more training data to `data/housing.csv` or adjust model hyperparameters in `model/train_model.py`
