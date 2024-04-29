#!/bin/bash

# Check for version tag argument
if [ -z "$1" ]; then
    echo "Error: Please provide a version tag as an argument."
    echo "Usage: ./build.sh <version_tag>"
    exit 1
fi

VERSION_TAG=$1
git clone https://github.com/arpg/GLIP


# Use the version tag in the Docker build command
DOCKER_BUILDKIT=0 docker build -t glip_server:$VERSION_TAG .
rm -rf GLIP
