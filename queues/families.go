package queues

import (
	"vulkan-tutorial/optional"
)

// FamilyIndices holds the indexes of Vulkan queue families needed by the programs.
type FamilyIndices struct {

	// GraphicsAndCompute is the index of the graphics queue family.
	GraphicsAndCompute optional.Optional[uint32]

	// Present is the index of the queue family used for presenting to the drawing
	// surface.
	Present optional.Optional[uint32]
}

// IsComplete returns true if all families have been set.
func (f *FamilyIndices) IsComplete() bool {
	return f.GraphicsAndCompute.HasValue() && f.Present.HasValue()
}
