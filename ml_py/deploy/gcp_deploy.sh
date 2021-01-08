#!/bin/bash
# set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
cd $DIR

source ./.env
IMAGE_NAME="ml-gallery-py"

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 $DIR
gcloud beta run deploy $IMAGE_NAME --image=gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 --allow-unauthenticated --memory=2048Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

# Set default region:
# gcloud config set run/region us-east4