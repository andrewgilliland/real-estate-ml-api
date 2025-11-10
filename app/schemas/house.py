from pydantic import BaseModel, Field
from typing import Optional


class HousePredictionRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: float = Field(..., ge=0.5, le=10.0)
    sqft_living: int = Field(..., ge=500, le=10000)
    sqft_lot: int = Field(..., ge=1000, le=50000)
    floors: float = Field(..., ge=1, le=4)
    zipcode: int = Field(..., ge=10000, le=99999)


class HousePredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    model_version: str
    trained_on: Optional[str] = None
    model_type: str = "RandomForestRegressor"
