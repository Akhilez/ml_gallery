#!/usr/bin/env bash

gcloud builds submit --tag gcr.io/everst-website/ml-gallery:v1 .
gcloud beta run deploy ml-gallery --image=gcr.io/everst-website/ml-gallery:v1 --platform managed --allow-unauthenticated --project everst-website

#gcloud config set run/region us-east4