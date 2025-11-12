from pandas import pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib
from pathlib import Path
import json
from datetime import datetime


def train_model():
    # Core Requirements for train_model.py:
    # 1. Data Loading & Preprocessing
    # Load the housing dataset from data/housing.csv
    print("Loading training data...")
    data_path = Path("data/housing.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)  # df = DataFrame
    print(f"Loaded {len(df)} records.")

    # Prepare features and target
    feature_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "zipcode",
    ]
    # Separate features (house characteristics) and target (price)
    features = df[feature_columns]
    prices = df["price"]

    # 2. Data Splitting
    # Split data into training and testing sets
    # Use 20% for testing, 80% for training
    # Ensure reproducibility with random state
    # With same random_state, both models train on identical data
    # Fair comparison of which algorithm is better
    # 42 = "The Answer to Life, Universe, and Everything" from Hitchhiker's Guide to the Galaxy
    train_features, test_features, train_prices, test_prices = train_test_split(
        features, prices, test_size=0.2, random_state=42
    )

    # 3. Model Training
    # Train a regression model (likely RandomForestRegressor or similar)

    # Random Forest Regressor - specific machine learning algorithm
    # "Random Forest" = Collection of Decision Trees
    # "Regressor" = Predicts Numbers
    # - Classification: Predicts categories (cat/dog, spam/not spam)
    # - Regression: Predicts continuous numbers (house prices, temperatures)

    # Real estate relationships are complex:
    # - Big house + bad zipcode = medium price
    # - Small house + great zipcode = high price
    # - Random Forest captures these patterns automatically
    model = RandomForestRegressor(
        n_estimators=100,  # 100 decision trees
        max_depth=10,  # Prevent overfitting
        random_state=42,  # Reproducible results
        n_jobs=-1,  # Use all CPU cores
    )

    # Fit the model to training data
    model.fit(train_features, train_prices)

    # Make predictions on test set
    predicted_prices = model.predict(test_features)

    # 4. Model Evaluation
    # Calculate R² score on test set
    r2 = r2_score(test_prices, predicted_prices)
    print(f"Model R² on test set: {r2:.4f}")
    # Target R² ≥ 0.85 on test data (per success metrics)
    if r2 >= 0.85:
        print("✅ Model meets accuracy target!")
    else:
        print("⚠️ Model needs improvement")

    # Generate other relevant metrics (MAE, RMSE)
    mae = mean_absolute_error(test_prices, predicted_prices)
    rmse = root_mean_squared_error(test_prices, predicted_prices, squared=False)
    print(f"Model MAE on test set: {mae:.4f}")
    print(f"Model RMSE on test set: {rmse:.4f}")

    # Log performance metrics

    # Compare training vs test performance
    training_r2 = model.score(train_features, train_prices)
    testing_r2 = model.score(test_features, test_prices)

    print(f"Training R²: {training_r2:.4f}")
    print(f"Testing R²: {testing_r2:.4f}")
    print(f"Gap: {training_r2 - testing_r2:.4f}")

    # Good: Gap < 0.05
    # Warning: Gap > 0.10
    # Overfitting: Gap > 0.15

    # 5. Model Persistence
    # Save the trained model to app/models/regressor.pkl using joblib
    # Save model metadata (version, training date, performance metrics)
    # Save the trained model
    model_dir = Path("app/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    # Save model file
    joblib.dump(model, model_dir / "regressor.pkl")

    # Save comprehensive metadata
    metadata = {
        "model_version": "1.0.0",
        "model_type": "RandomForestRegressor",
        "trained_on": datetime.now().isoformat(),
        "training_samples": len(df),
        "features": feature_columns,
        "performance": {
            "training_r2": float(training_r2),
            "testing_r2": float(testing_r2),
            "mae": float(mae),
            "rmse": float(rmse),
        },
        "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "feature_importance": dict(zip(feature_columns, model.feature_importances_)),
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# 6. Feature Schema Validation
# Ensure the model expects the same features defined in the API schema:
# bedrooms, bathrooms, sqft_living, sqft_lot, floors, zipcode


# 7. Preprocessing Pipeline
# Handle categorical variables (like zipcode)
# Scale numerical features if needed
# Create a preprocessing pipeline that can be reused in the API


# 8. Model Versioning
# Set model version (referenced in /version endpoint)
# Track training timestamp
# The file should be executable as a standalone script and integrate with the data preprocessing utilities in app/utils/preprocess.py to ensure consistency between training and prediction phases.
