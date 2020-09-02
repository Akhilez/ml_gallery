#!/bin/bash

wget https://storage.googleapis.com/akhilez/datasets/singularity_systems/data.tgz
rm -rf data
tar -xzf data.tgz
find data -type f -exec mv '{}' '{}'.txt \;

mkdir trainers/models
