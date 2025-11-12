from pydantic import BaseModel, Field
from typing import Optional, List


class TrainingRequest(BaseModel):
    retrain: bool = Field(default=True, description="Whether to retrain the model")
    model_version: Optional[str] = Field(default=None, description="New model version")


class TrainingResponse(BaseModel):
    status: str = Field(..., description="Training status")
    message: str = Field(default=None, description="Additional information")
    model_version: str = Field(default=None, description="Trained model version")
    performance_metrics: dict = Field(
        default=None, description="Performance metrics of the trained model"
    )
    training_duration_seconds: float = Field(..., description="How long training was")


class TrainingStatus(BaseModel):
    is_training: bool = Field(..., description="Indicates if training is in progress")
    current_model_version: str = Field(..., description="Current model version")
    last_trained: Optional[str] = Field(None, description="Timestamp of last training")
