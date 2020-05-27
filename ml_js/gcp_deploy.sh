#!/usr/bin/env bash

PROJECT_ID="graphic-jet-278213"
IMAGE_NAME="ml-gallery"

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 $DIR

gcloud beta run deploy $IMAGE_NAME --image=gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

# gcloud builds submit --tag gcr.io/graphic-jet-278213/ml-gallery:v1 .
# gcloud beta run deploy ml-gallery --image=gcr.io/graphic-jet-278213/ml-gallery:v1 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

#gcloud config set run/region us-east4
