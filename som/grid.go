package som

import "github.com/gonum/matrix/mat64"

// RandInit returns a matrix with number of rows supplied as a parameter
// and ithe same number of columns as a matrix passed in as argument.
// The returned matris is initialized to unformly distributed random values
// in range from 0 to 1. It fails with error if the new matrix could not be initialized
func RandInit(inMx mat64.Matrix, rows int) (*mat64.Dense, error) {
	return nil, nil
}

// LinInit returns a matrix with number of rows supplied as a parameter
// and ithe same number of columns as a matrix passed in as argument.
// The returned matris is initialized to values from a linear space
// spanned by the values stored in the matrix supplied as a parameter.
// It fails with error if the new matrix could not be initialized.
func LinInit(inMx mat64.Matrix, rows int) (*mat64.Dense, error) {
	return nil, nil
}

// PlaneGridCoords returns a matrix that contains planar coordinates of SOM units
// stored row by row. Planar coordinates naturally imply two columns.
// It fails with error if the requested unit shape is unsupported or
// if the matrix fails to be initialized
func PlaneGridCoords(uShape string) (*mat64.Dense, error) {
	return nil, nil
}
