import json
import os
from pathlib import Path
from mangum import Mangum
import boto3
import joblib

from app.main import app

# Configuration
S3_BUCKET = os.environ.get("MODEL_BUCKET", "real-estate-ml-models")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/regressor.pkl")
METADATA_KEY = os.environ.get("METADATA_KEY", "models/metadata.json")

# Global variables for Lambda reuse
model = None
metadata = {}
s3_client = None


def load_model_from_s3():
    """Download and load model from S3 on cold start"""
    global model, metadata, s3_client

    if model is not None:
        return  # Model already loaded

    print("Cold start - loading model from S3...")

    # Initialize S3 client
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Download model
    model_path = Path("/tmp/regressor.pkl")
    metadata_path = Path("/tmp/metadata.json")

    try:
        # Download model file
        s3_client.download_file(S3_BUCKET, MODEL_KEY, str(model_path))
        model = joblib.load(model_path)
        print("✅ Model loaded from S3")

        # Download metadata
        s3_client.download_file(S3_BUCKET, METADATA_KEY, str(metadata_path))
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded: v{metadata.get('model_version', 'unknown')}")

    except Exception as e:
        print(f"❌ Error loading model from S3: {e}")
        raise


# Load model on cold start (outside handler for reuse)
load_model_from_s3()

# Update app's global model variable
app.state.model = model
app.state.metadata = metadata

# Create Lambda handler
handler = Mangum(app, lifespan="off")
