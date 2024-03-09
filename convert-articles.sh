#!/usr/bin/env bash

# This code was used for initial conversion of the articles from the original
# repository.

sed -i -E 's/00_base_code.cpp/00_base\/main.go/' \
    en/03_Drawing_a_triangle/00_Setup/00_Base_code.md

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\(\/code\/(.+)\.cpp/\(\/code\/\1\/main.go/' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/C\+\+ code/Go code/' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/09_shader_base/\/code\/09_shader_modules\/shaders\/triangle/' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/18_shader_vertexbuffer/\/code\/18_vertex_input\/shaders\/shader/' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/22_shader_ubo/\/code\/22_descriptor_set_layout\/shaders\/shader/' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/26_shader_textures/\/code\/26_texture_mapping\/shaders\/shader/' {} \;

sed -i -E 's/code\/27_depth_buffering\/main.go/code\/27_shader_depth\/main.go/' \
    en/07_Depth_buffering.md

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/27_shader_depth\./\/code\/27_shader_depth\/shaders\/shader./' {} \;

find en/ -type f -name '*.md' -exec \
    sed -i -E 's/\/code\/31_shader_compute\./\/code\/31_shader_compute\/shaders\/shader./' {} \;

sed -i -E 's/code\/31_compute_shader\/main.go/code\/31_shader_compute\/main.go/' \
    en/11_Compute_Shader.md
