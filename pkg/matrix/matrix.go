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
	return withValidCols(cols, m, mat64.Max)
}

// ColsMin returns a slice of min values of first cols number of matrix columns
// It returns error if passed in matrix is nil, has zero size or requested number
// of columns exceeds the number of columns in the matrix passed in as parameter.
func ColsMin(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidCols(cols, m, mat64.Min)
}

// ColsMean returns a slice of mean values of first cols number of matrix columns
// It returns error if passed in matrix is nil, has zero size or requested number
// of columns exceeds the number of columns in the matrix passed in as parameter.
func ColsMean(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidCols(cols, m, mean)
}

// ColsStdev returns a slice of standard deviatioasn of first cols number of matrix columns
// It returns error if passed in matrix is nil, has zero size or requested number
// of columns exceeds the number of columns in the matrix passed in as parameter.
func ColsStdev(cols int, m *mat64.Dense) ([]float64, error) {
	return withValidCols(cols, m, stdev)
}

// RowsMax returns a slice of max values of first rows number of matrix rows.
// It returns error if passed in matrix is nil, has zero size or requested number
// of rows exceeds the number of rows in the matrix passed in as parameter.
func RowsMax(rows int, m *mat64.Dense) ([]float64, error) {
	return withValidRows(rows, m, mat64.Max)
}

// RowsMin returns a slice of min values of first rows number of matrix rows.
// It returns error if passed in matrix is nil, has zero size or requested number
// of rows exceeds the number of rows in the matrix passed in as parameter.
func RowsMin(rows int, m *mat64.Dense) ([]float64, error) {
	return withValidRows(rows, m, mat64.Min)
}

// MakeRandom creates a new matrix rows x cols matrix which is initialized
// to random numbers uniformly distributed in interval [min, max].
// MakeRandom fails if invalid matrix dimensions are requested.
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
// It modifies the matrix passed in.
// AddConstant fails with error if empty matrix is supplied
func AddConst(val float64, m *mat64.Dense) (*mat64.Dense, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
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

// colsFn applies function fn on each column, collects the results into slice and returns it
func colsFn(cols int, m *mat64.Dense, fn func(mat64.Matrix) float64) []float64 {
	res := make([]float64, cols)
	for i := 0; i < cols; i++ {
		res[i] = fn(m.ColView(i))
	}
	return res
}

// rowsFn applies function fn on each row, collects the results into slice and returns it
func rowsFn(rows int, m *mat64.Dense, fn func(mat64.Matrix) float64) []float64 {
	res := make([]float64, rows)
	for i := 0; i < rows; i++ {
		res[i] = fn(m.RowView(i))
	}
	return res
}

// withValidCols validates that the passed in matrix is non-nil, has non-zero number of columns
// and executes the fn function on count number of matrix columns. It collects the results of
// each calculation and returns it in a slice.
// It returns error if either the passed in matrix is nil, has zero size or requested number of
// columns is larger than the number of matrix columns.
func withValidCols(count int, m *mat64.Dense, fn func(mat64.Matrix) float64) ([]float64, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
	}
	_, cols := m.Dims()
	if cols == 0 {
		return nil, fmt.Errorf("Invalid number of columns supplied: %v\n", m)
	}
	if count > cols {
		return nil, fmt.Errorf("Column count exceeds matrix dimensions: %d\n", count)
	}
	return colsFn(count, m, fn), nil
}

// withValidRows validates that the passed in matrix is non-nil, has non-zero number of rows
// and executes the fn function on count number of matrix rows. It collects the results of
// each calculation and returns it in a slice.
// It returns error if either the passed in matrix is nil, has zero size or requested number of
// rows is larger than the number of matrix rows.
func withValidRows(count int, m *mat64.Dense, fn func(mat64.Matrix) float64) ([]float64, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
	}
	rows, _ := m.Dims()
	if rows == 0 {
		return nil, fmt.Errorf("Invalid number of rows supplied: %v\n", m)
	}
	if count > rows {
		return nil, fmt.Errorf("Row count exceeds matrix dimensions: %d\n", count)
	}
	return rowsFn(count, m, fn), nil
}

// withValidDims validates the requested rows and cols matrix dimensions
// It treturns error if either rows or cols are non-positive integers
func withValidDims(rows, cols int, fn func() (*mat64.Dense, error)) (*mat64.Dense, error) {
	// can not create matrix with negative dimensions
	if rows <= 0 {
		return nil, fmt.Errorf("Invalid number of rows: %d\n", rows)
	}
	if cols <= 0 {
		return nil, fmt.Errorf("Invalid number of columns: %d\n", cols)
	}
	return fn()
}
