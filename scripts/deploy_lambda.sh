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

# Create S3 bucket if it doesn't exist
echo "1️⃣ Checking S3 bucket..."
if ! aws s3 ls "s3://$BUCKET_NAME" 2>&1 > /dev/null; then
    echo "   Creating bucket $BUCKET_NAME..."
    aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled \
        --region "$AWS_REGION"
    
    # Block public access
    aws s3api put-public-access-block \
        --bucket "$BUCKET_NAME" \
        --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" \
        --region "$AWS_REGION"
    
    echo "   ✅ Bucket created and configured"
else
    echo "   ✅ Bucket already exists"
fi

# Upload trained model to S3
echo "2️⃣ Uploading trained model..."
if [ ! -f "app/models/regressor.pkl" ]; then
    echo "   ❌ Model not found. Run: uv run python model/train_model.py"
    exit 1
fi
aws s3 cp app/models/regressor.pkl "s3://$BUCKET_NAME/models/regressor.pkl"
aws s3 cp app/models/metadata.json "s3://$BUCKET_NAME/models/metadata.json"
echo "   ✅ Model uploaded"

# Create Lambda layer for scikit-learn
echo "3️⃣ Creating Lambda layer for ML libraries..."
rm -rf layer/
mkdir -p layer/python
pip install \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    joblib==1.4.2 \
    -t layer/python \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --python-version "$PYTHON_VERSION"

cd layer
zip -r ../sklearn-layer.zip python > /dev/null
cd ..
echo "   ✅ Layer package created"

# Package Lambda function
echo "4️⃣ Packaging Lambda function..."
rm -rf package/
mkdir -p package
cp -r app/ package/

# Install FastAPI dependencies
pip install \
    fastapi==0.115.6 \
    mangum==0.19.0 \
    pydantic==2.10.4 \
    boto3==1.35.80 \
    -t package \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --python-version "$PYTHON_VERSION"

cd package
zip -r ../lambda-package.zip . > /dev/null
cd ..
echo "   ✅ Lambda package created"

# Upload Lambda artifacts to S3
echo "5️⃣ Uploading Lambda artifacts..."
aws s3 cp sklearn-layer.zip "s3://$BUCKET_NAME/layers/sklearn-layer.zip"
aws s3 cp lambda-package.zip "s3://$BUCKET_NAME/lambda-package.zip"
rm sklearn-layer.zip lambda-package.zip
echo "   ✅ Artifacts uploaded"

# Deploy CloudFormation stack
echo "6️⃣ Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file cloudformation/template.yaml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides ModelBucketName="$BUCKET_NAME" \
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
