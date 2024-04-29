#!/bin/bash

# Check for version tag argument
if [ -z "$1" ]; then
    echo "Error: Please provide a version tag as an argument."
    echo "Usage: ./build.sh <version_tag>"
    exit 1
fi

VERSION_TAG=$1

# Use the version tag in the Docker build command
DOCKER_BUILDKIT=0 docker build -t yolo_world_server:$VERSION_TAG .
rm -rf GLIP
