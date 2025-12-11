#!/bin/bash

set -e

# Configuration
STACK_NAME="real-estate-ml-api"
BUCKET_NAME="real-estate-ml-models-$(aws sts get-caller-identity --query Account --output text)"
AWS_REGION="${AWS_REGION:-us-east-1}"
PYTHON_VERSION="3.11"

echo "🚀 Starting Lambda deployment..."
echo "📦 Bucket: $BUCKET_NAME"
echo "🌍 Region: $AWS_REGION"

# Upload trained model to S3
echo "1️⃣ Checking trained model..."
if [ ! -f "app/models/regressor.pkl" ]; then
    echo "   ❌ Model not found. Run: uv run python model/train_model.py"
    exit 1
fi
echo "   ✅ Model found"

# Create Lambda layer for scikit-learn
echo "2️⃣ Creating Lambda layer for scikit-learn only..."
rm -rf layer/
mkdir -p layer/python
uv pip install \
    scikit-learn==1.5.2 \
    --target layer/python \
    --python-platform x86_64-manylinux2014 \
    --python-version "$PYTHON_VERSION"

cd layer
zip -r ../sklearn-layer.zip python > /dev/null
cd ..
echo "   ✅ Layer package created"

# Package Lambda function
echo "3️⃣ Packaging Lambda function..."
rm -rf package/
mkdir -p package
cp -r app/ package/

# Install FastAPI dependencies + ML libraries (pandas, numpy, joblib)
uv pip install \
    fastapi==0.115.6 \
    mangum==0.19.0 \
    pydantic==2.10.4 \
    boto3==1.35.80 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    joblib==1.4.2 \
    --target package \
    --python-platform x86_64-manylinux2014 \
    --python-version "$PYTHON_VERSION"

cd package
zip -r ../lambda-package.zip . > /dev/null
cd ..
echo "   ✅ Lambda package created"

# Deploy CloudFormation stack (DeployLambdaResources=false first to create bucket)
echo "4️⃣ Deploying CloudFormation stack (creating S3 bucket)..."
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides ModelBucketName="$BUCKET_NAME" DeployLambdaResources="false" \
    --capabilities CAPABILITY_IAM \
    --region "$AWS_REGION"

echo "   ✅ Bucket created"

# Upload artifacts to S3
echo "5️⃣ Uploading artifacts to S3..."
aws s3 cp app/models/regressor.pkl "s3://$BUCKET_NAME/models/regressor.pkl"
aws s3 cp app/models/metadata.json "s3://$BUCKET_NAME/models/metadata.json"
aws s3 cp sklearn-layer.zip "s3://$BUCKET_NAME/layers/sklearn-layer.zip"
aws s3 cp lambda-package.zip "s3://$BUCKET_NAME/lambda-package.zip"
rm sklearn-layer.zip lambda-package.zip
echo "   ✅ Artifacts uploaded"

# Update stack to deploy Lambda resources
echo "6️⃣ Updating CloudFormation stack (deploying Lambda resources)..."
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides ModelBucketName="$BUCKET_NAME" DeployLambdaResources="true" \
    --capabilities CAPABILITY_IAM \
    --region "$AWS_REGION"

# Get API URL
echo "7️⃣ Retrieving API URL..."
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text \
    --region "$AWS_REGION")

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📍 API URL: $API_URL"
echo ""
echo "Test endpoints:"
echo "  Health:  curl $API_URL/health"
echo "  Version: curl $API_URL/version"
echo "  Predict: curl -X POST $API_URL/predict -H 'Content-Type: application/json' -d '{\"bedrooms\": 3, \"bathrooms\": 2, \"sqft_living\": 2000, \"sqft_lot\": 5000, \"floors\": 2, \"zipcode\": 98052}'"
echo ""
