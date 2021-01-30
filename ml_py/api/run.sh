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

echo Running project $project

cd $project

uvicorn main:app --reload --port 8001
