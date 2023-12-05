package models

import "embed"

// FS contains all the models used throughout the examples. It makes it possible
// to generate a binary and just copy it to another machine.
//
//go:embed viking_room.obj
var FS embed.FS
