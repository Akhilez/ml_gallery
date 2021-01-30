#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

project=$1

if [[ $project == "" ]]; then
  echo No project name specified.
  exit 1
elif [[ ! -f $project/main.py ]]; then
  echo Invalid project
  exit 2
fi

echo Deploying project $project

source .env

# 1. Copy dockerfile to project directory
cp ./Dockerfile $project/

# 2. Update dockerfile: a. Change project name.
# sed -i "s/PROJECT/$project/g" $project/Dockerfile

cd $project

echo --------- Building image -----------
if [[ $2 == "local" ]]; then

  # 3. Build
  docker build -t $project:v0 .

  # 4. Deploy
  echo --------- Deploying -----------
  docker run -d -p 80:80 $project:v0

else

  # 3. Build
  gcloud builds submit --tag gcr.io/$GCP_PROJECT/${project}:v0 .

  # 4. Deploy
  echo --------- Deploying -----------
  service_name=${project/_/-}  # Because underscores are not allowed
  gcloud beta run deploy $service_name --image=gcr.io/$GCP_PROJECT/$project:v0 --allow-unauthenticated --memory=2048Mi --timeout=900 --platform managed

fi

# 5. Delete dockerfile
rm ./Dockerfile
