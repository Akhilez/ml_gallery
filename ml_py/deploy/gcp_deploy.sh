#!/bin/bash
set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

source ../.env

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 $DIR

gcloud beta run deploy $CLOUD_RUN_SERVICE --image=gcr.io/$PROJECT_ID/$IMAGE_NAME:v1 --allow-unauthenticated --memory=2048Mi --timeout=900 --platform managed --set-env-vars=DEBUG=False

# WINDOWS:
# gcloud builds submit --tag gcr.io/lateral-imagery-267008/plabmaster_web:v1 .
# gcloud beta run deploy plabmaster-web --image=gcr.io/lateral-imagery-267008/plabmaster_web:v1 --allow-unauthenticated --memory=512Mi --timeout=900 --platform managed

# Set default region:
# gcloud config set run/region asia-east1