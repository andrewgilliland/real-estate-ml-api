# Core Requirements for train_model.py:

# 1. Data Loading & Preprocessing
# Load the housing dataset from data/housing.csv
# Handle missing values and data cleaning
# Feature engineering (if needed)
# Split features (X) from target variable (y - house prices)


# 2. Data Splitting
# Split data into training and testing sets
# Ensure reproducibility with random state


# 3. Model Training
# Train a regression model (likely RandomForestRegressor or similar)
# Target R² ≥ 0.85 on test data (per success metrics)


# 4. Model Evaluation
# Calculate R² score on test set
# Generate other relevant metrics (MAE, RMSE)
# Log performance metrics


# 5. Model Persistence
# Save the trained model to app/models/regressor.pkl using joblib
# Save model metadata (version, training date, performance metrics)


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
