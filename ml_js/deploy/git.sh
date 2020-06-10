#! /bin/bash

if [[ "$1" == "" ]]; then
    git status
    echo "No commit message provided. Usage: 'git.sh \"<commit_message>\"'"
    exit 1
fi

echo Commit message: "$1"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
echo "PWD: `pwd`"

git status
git add -A
git status
git commit -m "$1"
git push
