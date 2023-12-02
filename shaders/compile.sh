#!/usr/bin/env bash

glslc shader.frag -o frag.spv
glslc shader.vert -o vert.spv
glslc shader.comp -o comp.spv
