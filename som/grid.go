package som

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/esemsch/gosom/pkg/matrix"
)

// RandInit returns a matrix initialized to uniformly distributed random values
// in each column in range between [max, min] where max and min are maximum and minmum values
// in particular matrix column. The returned matrix has r number of rows and
// as many columns as the matrix passed in as a parameter.
// It fails with error if the new matrix could not be initialized or if inMx is nil.
func RandInit(inMx *mat64.Dense, rows int) (*mat64.Dense, error) {
	// if nil matrix is passed in, return error
	if inMx == nil {
		return nil, fmt.Errorf("Invalid input matrix: %v\n", inMx)
	}
	// input matrix dimensions
	_, cols := inMx.Dims()
	// get max of each column
	max, err := matrix.ColsMax(cols, inMx)
	if err != nil {
		return nil, err
	}
	// get min of each column
	min, err := matrix.ColsMin(cols, inMx)
	if err != nil {
		return nil, err
	}
	// initialize matrix to rand values between 0.0 and 1.0
	outMx, err := matrix.MakeRandom(rows, cols, 0.0, 1.0)
	if err != nil {
		return nil, err
	}
	for i := 0; i < cols; i++ {
		col := outMx.ColView(i)
		for j := 0; j < rows; j++ {
			val := col.At(j, 0)
			col.SetVec(j, val*(max[i]-min[i])+min[i])
		}
	}
	return outMx, nil
}

// LinInit returns a matrix initialized to values lying in a linear space
// spanned by the top principal components of the values stored in the matrix passed in
// as a parameter. The returned matrix has rows number of rows and
// as many columns as the input matrix passed in as a parameter
// It fails with error if the new matrix could not be initialized or if inMx is nil.
func LinInit(inMx *mat64.Dense, rows int) (*mat64.Dense, error) {
	// if nil matrix is passed in, return error
	if inMx == nil {
		return nil, fmt.Errorf("Invalid input matrix: %v\n", inMx)
	}
	return nil, nil
}

// PlaneGridCoords returns a matrix that contains planar coordinates of SOM units
// stored row by row. Planar coordinates naturally imply two columns.
// It fails with error if the requested unit shape is unsupported or
// if the matrix fails to be initialized
func PlaneGridCoords(uShape string) (*mat64.Dense, error) {
	return nil, nil
}
