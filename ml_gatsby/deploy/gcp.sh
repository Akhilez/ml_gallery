#!/usr/bin/env bash
# set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
cd $DIR
echo "PWD: `pwd`"

# # Build the project
npm run clean
npm run build

# # Setup docker
cp deploy/Dockerfile public/
cp deploy/nginx.conf public/
cd public

# Deployment
PROJECT_ID="graphic-jet-278213"
IMAGE_NAME="mlg"

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 `pwd`
gcloud beta run deploy $IMAGE_NAME --image=gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 --port=80 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed

#gcloud config set run/region us-east4
