#! /bin/bash

if [[ "$1" == "" ]]; then
    git status
    echo "No commit message provided. Usage: 'all.sh \"<commit_message>\"'"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
echo "PWD: `pwd`"

sh git.sh $1
sh gcp.sh
