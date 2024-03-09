#!/usr/bin/env bash

docker run --rm -it -v $(pwd):/build/docs/ overv/vulkan-tutorial \
    daux generate --delete \
        -c /build/docs/config.json \
        -s /build/docs/ \
        -d /build/docs/out/
