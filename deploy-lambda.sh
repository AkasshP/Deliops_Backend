#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${ECR_REPOSITORY:=deliops-backend}"
: "${LAMBDA_FUNCTION_NAME:=deliops-backend}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -t "${ECR_REPOSITORY}:latest" .
docker tag "${ECR_REPOSITORY}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

IMAGE_URI="${ECR_URI}:latest"
if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$LAMBDA_FUNCTION_NAME" --image-uri "$IMAGE_URI" --region "$AWS_REGION"
else
  echo "Create the Lambda execution role first, then run:"
  echo "aws lambda create-function --function-name ${LAMBDA_FUNCTION_NAME} --package-type Image --code ImageUri=${IMAGE_URI} --role <lambda-execution-role-arn> --timeout 30 --memory-size 1024 --region ${AWS_REGION}"
  exit 1
fi
