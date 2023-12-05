package textures

import "embed"

// FS contains all the textures used throughout the examples. It makes it possible
// to generate a binary and just copy it to another machine.
//
//go:embed viking_room.png
//go:embed texture.jpg
var FS embed.FS
