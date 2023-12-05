package shaders

import "embed"

//go:generate glslc shader.frag -o frag.spv
//go:generate glslc shader.vert -o vert.spv
//go:generate glslc shader.comp -o comp.spv

// FS embed the vertex and fragnent shaders. Run `go generate` in order to compile
// them again.
//
//go:embed frag.spv
//go:embed vert.spv
//go:embed comp.spv
var FS embed.FS
