package matrix

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// ColsMax returns a slice of max values per each matrix column
// It returns error if passed in matrix is nil or has zero size
func ColsMax(m *mat64.Dense) ([]float64, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
	}
	_, cols := m.Dims()
	if cols == 0 {
		return nil, fmt.Errorf("Empty matrix supplied: %v\n", m)
	}
	max := make([]float64, cols)
	for i := 0; i < cols; i++ {
		max[i] = mat64.Max(m.ColView(i))
	}
	return max, nil
}

// ColsMin returns a slice of min values per each matrix column
// It returns error if passed in matrix is nil or has zero size
func ColsMin(m *mat64.Dense) ([]float64, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
	}
	_, cols := m.Dims()
	if cols == 0 {
		return nil, fmt.Errorf("Empty matrix supplied: %v\n", m)
	}
	min := make([]float64, cols)
	for i := 0; i < cols; i++ {
		min[i] = mat64.Min(m.ColView(i))
	}
	return min, nil
}

// MakeRandom creates a new matrix rows x cols matrix which is initialized
// to random numbers uniformly distributed in interval [min, max].
// MakeRandom fails if invalid matrix dimensions are requested.
func MakeRandom(rows, cols int, min, max float64) (*mat64.Dense, error) {
	// can not create matrix with negative dimensions
	if rows <= 0 {
		return nil, fmt.Errorf("Invalid number of rows: %d\n", rows)
	}
	if cols <= 0 {
		return nil, fmt.Errorf("Invalid number of columns: %d\n", cols)
	}
	// set random seed
	rand.Seed(55)
	// allocate data slice
	randVals := make([]float64, rows*cols)
	for i := range randVals {
		// we need value between 0 and 1.0
		randVals[i] = rand.Float64()*(max-min) + min
	}
	return mat64.NewDense(rows, cols, randVals), nil
}

// MakeConstant returns a matrix of rows x cols whose each element is set to val.
// MakeConstant fails if invalid matrix dimensions are requested.
func MakeConstant(rows, cols int, val float64) (*mat64.Dense, error) {
	// can not create matrix with negative dimensions
	if rows <= 0 {
		return nil, fmt.Errorf("Invalid number of rows: %d\n", rows)
	}
	if cols <= 0 {
		return nil, fmt.Errorf("Invalid number of columns: %d\n", cols)
	}
	// allocate zero matrix and set every element to val
	constMx := mat64.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			constMx.Set(i, j, val)
		}
	}
	return constMx, nil
}
