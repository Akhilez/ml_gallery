#!/usr/bin/env bash
# set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
cd $DIR
echo "PWD: `pwd`"

# Build the project
npm run clean
npm run build

# Setup docker
cp deploy/Dockerfile public/
cp deploy/nginx.conf public/
cd public

# Deployment
IMAGE_NAME="mlg"

docker build -t $IMAGE_NAME:v1 `pwd`
docker run --publish 8000:8000 $IMAGE_NAME:v1
