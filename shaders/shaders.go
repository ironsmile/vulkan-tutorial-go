package shaders

import "embed"

//go:generate ./compile.sh

// FS embed the vertex and fragnent shaders. Run `go generate` in order to compile
// them again.
//
//go:embed frag.spv
//go:embed vert.spv
//go:embed comp.spv
var FS embed.FS
