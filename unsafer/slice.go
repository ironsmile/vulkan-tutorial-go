package unsafer

import (
	"reflect"
	"unsafe"
)

// SliceToBytes interprets an arbitrary input slice as a byte slice.
//
// Note that the returned slice points to the same underlying data in memory. It
// does not make a copy.
func SliceToBytes[T any](input []T) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&input))
	header.Len = int(unsafe.Sizeof(input[0])) * len(input)
	header.Cap = header.Len
	bytesSlice := *(*[]byte)(unsafe.Pointer(&header))
	return bytesSlice
}
