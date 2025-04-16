#!/bin/bash

if [ "$1" == "build" ]; then
    docker rm -f dimos-dev
    docker build \
        --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
        --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
        -t dimensionalos/dev-base docker/dev/base/
fi

docker-compose -f docker/dev/base/docker-compose.yaml up -d && docker exec -it dimos-dev bash
