#!/bin/bash
set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
cd $DIR

docker build -t py-ml-gallery:v1 $DIR

docker run --publish 8001:8001 py-ml-gallery:v1

# docker ps -a
# docker rm <container id> [<container id>,]

# docker images
# docker rmi <image id> [<image id>,]
