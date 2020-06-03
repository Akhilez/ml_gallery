#!/usr/bin/env bash
# set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
echo "PWD: `pwd`"

PROJECT_ID="graphic-jet-278213"
IMAGE_NAME="ml-gallery"

echo gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 $DIR

echo gcloud beta run deploy $IMAGE_NAME --image=gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

# gcloud builds submit --tag gcr.io/graphic-jet-278213/ml-gallery:v1 .
# gcloud beta run deploy ml-gallery --image=gcr.io/graphic-jet-278213/ml-gallery:v1 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

#gcloud config set run/region us-east4
