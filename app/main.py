from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.schemas.house import (
    HousePredictionRequest,
    HousePredictionResponse,
    HealthResponse,
    VersionResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate ML API",
    description="A lightweight FastAPI-based backend service that predicts residential property prices",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use app.state for Lambda compatibility
app.state.model = None
app.state.metadata = {}


@app.on_event("startup")
async def load_model():
    """Load the trained model and metadata on startup"""

    model_path = Path("app/models/regressor.pkl")
    metadata_path = Path("app/models/metadata.json")

    try:
        if model_path.exists():
            app.state.model = joblib.load(model_path)
            print("✅ Model loaded successfully")
        else:
            print(
                "⚠️  Model file not found. Train a model first using 'uv run python model/train_model.py'"
            )

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                app.state.metadata = json.load(f)
            print(
                f"✅ Model metadata loaded: v{app.state.metadata.get('model_version', 'unknown')}"
            )

    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.post("/predict", response_model=HousePredictionResponse)
async def predict_price(request: HousePredictionRequest):
    """Predict house price based on input features"""

    if app.state.model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train a model first."
        )

    try:
        # Convert request to DataFrame
        features = pd.DataFrame(
            [
                {
                    "bedrooms": request.bedrooms,
                    "bathrooms": request.bathrooms,
                    "sqft_living": request.sqft_living,
                    "sqft_lot": request.sqft_lot,
                    "floors": request.floors,
                    "zipcode": request.zipcode,
                }
            ]
        )

        # Make prediction
        prediction = app.state.model.predict(features)[0]

        return HousePredictionResponse(
            predicted_price=round(float(prediction), 2),
            model_version=app.state.metadata.get("model_version", "unknown"),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok" if app.state.model is not None else "degraded")


@app.get("/version", response_model=VersionResponse)
async def get_version():
    """Get model version and metadata"""
    return VersionResponse(
        model_version=app.state.metadata.get("model_version", "unknown"),
        trained_on=app.state.metadata.get("trained_on"),
        model_type="RandomForestRegressor",
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Real Estate ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "version": "GET /version",
        },
    }
