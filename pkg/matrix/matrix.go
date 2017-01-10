package matrix

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// ColsMax returns a slice of max values of first cols number of matrix columns
// It returns error if passed in matrix is nil, has zero size or requested number
// of columns exceeds the number of columns in the matrix passed in as parameter.
func ColsMax(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("cols", cols, m, mat64.Max)
}

// ColsMin returns a slice of min values of first cols number of matrix columns
// It returns error if passed in matrix is nil, has zero size or requested number
// of columns exceeds the number of columns in the matrix passed in as parameter.
func ColsMin(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("cols", cols, m, mat64.Min)
}

// ColsMean returns a slice of mean values of first cols matrix columns
// It returns error if passed in matrix is nil or has zero size or requested number
// of columns exceeds the number of columns in matrix m.
func ColsMean(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("cols", cols, m, mean)
}

// ColsStdev returns a slice of standard deviations of first cols matrix columns
// It returns error if passed in matrix is nil or has zero size or requested number
// of columns exceeds the number of columns in matrix m.
func ColsStdev(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("cols", cols, m, stdev)
}

// RowsMax returns a slice of max values of first rows matrix rows.
// It returns error if passed in matrix is nil or has zero size or requested number
// of rows exceeds the number of rows in matrix m.
func RowsMax(rows int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("rows", rows, m, mat64.Max)
}

// RowsMin returns a slice of min values of first rows matrix rows.
// It returns error if passed in matrix is nil or has zero size or requested number
// of rows exceeds the number of rows in matrix m.
func RowsMin(rows int, m *mat64.Dense) ([]float64, error) {
	return withValidDim("rows", rows, m, mat64.Min)
}

// MakeRandom creates a new matrix with provided number of rows and columns
// which is initialized to random numbers uniformly distributed in interval [min, max].
// MakeRandom fails if non-positive matrix dimensions are requested.
func MakeRandom(rows, cols int, min, max float64) (*mat64.Dense, error) {
	return withValidDims(rows, cols, func() (*mat64.Dense, error) {
		// set random seed
		rand.Seed(55)
		// allocate data slice
		randVals := make([]float64, rows*cols)
		for i := range randVals {
			// we need value between 0 and 1.0
			randVals[i] = rand.Float64()*(max-min) + min
		}
		return mat64.NewDense(rows, cols, randVals), nil
	})
}

// MakeConstant returns a matrix of rows x cols whose each element is set to val.
// MakeConstant fails if invalid matrix dimensions are requested.
func MakeConstant(rows, cols int, val float64) (*mat64.Dense, error) {
	return withValidDims(rows, cols, func() (*mat64.Dense, error) {
		// allocate zero matrix and set every element to val
		constMx := mat64.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				constMx.Set(i, j, val)
			}
		}
		return constMx, nil
	})
}

// AddConst adds a constant value to every element of matrix
// It modifies the matrix m passed in as a paramter.
// AddConstant fails with error if empty matrix is supplied
func AddConst(val float64, m *mat64.Dense) (*mat64.Dense, error) {
	if m == nil {
		return nil, fmt.Errorf("invalid matrix supplied: %v", m)
	}
	rows, cols := m.Dims()
	return withValidDims(rows, cols, func() (*mat64.Dense, error) {
		// allocate zero matrix and set every element to val
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				m.Set(i, j, m.At(i, j)+val)
			}
		}
		return m, nil
	})
}

// viewFunc defines matrix dimension view function
type viewFunc func(int) *mat64.Vector

// dimFn applies function fn to first count matrix rows or columns.
// dim can be either set to rows or cols.
// dimFn collects the results into a slice and returns it
func dimFn(dim string, count int, m *mat64.Dense, fn func(mat64.Matrix) float64) []float64 {
	res := make([]float64, count)
	var viewFn viewFunc
	switch dim {
	case "rows":
		viewFn = m.RowView
	case "cols":
		viewFn = m.ColView
	}
	for i := 0; i < count; i++ {
		res[i] = fn(viewFn(i))
	}
	return res
}

// withValidDim executes function fn on first count of matrix columns or rows.
// It collects the results of each calculation and returns it in a slice.
// It returns error if either matrix m is nil, has zero size or requested number of
// particular dimension is larger than the matrix m dimensions.
func withValidDim(dim string, count int, m *mat64.Dense,
	fn func(mat64.Matrix) float64) ([]float64, error) {
	// matrix can't be nil
	if m == nil {
		return nil, fmt.Errorf("invalid matrix supplied: %v", m)
	}
	rows, cols := m.Dims()
	switch dim {
	case "rows":
		if rows == 0 {
			return nil, fmt.Errorf("invalid number of rows supplied: %v", m)
		}
		if count > rows {
			return nil, fmt.Errorf("row count exceeds matrix rows: %d", count)
		}
	case "cols":
		if cols == 0 {
			return nil, fmt.Errorf("invalid number of columns supplied: %v", m)
		}
		if count > cols {
			return nil, fmt.Errorf("column count exceeds matrix columns: %d", count)
		}
	}
	return dimFn(dim, count, m, fn), nil
}

// withValidDims validates if the rows and cols are valid matrix dimensions
// It returns error if either rows or cols are invalid i.e. non-positive integers
func withValidDims(rows, cols int, fn func() (*mat64.Dense, error)) (*mat64.Dense, error) {
	// can not create matrix with negative dimensions
	if rows <= 0 {
		return nil, fmt.Errorf("invalid number of rows: %d", rows)
	}
	if cols <= 0 {
		return nil, fmt.Errorf("invalid number of columns: %d", cols)
	}
	return fn()
}

// returns a mean valur for a given matrix
func mean(m mat64.Matrix) float64 {
	r, c := m.Dims()
	return mat64.Sum(m) / (float64(r) * float64(c))
}

// returns a mean valur for a given matrix
func stdev(m mat64.Matrix) float64 {
	r, _ := m.Dims()
	col := make([]float64, r)
	mat64.Col(col, 0, m)
	return stat.StdDev(col, nil)
}
