package som

import (
	"fmt"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestGridDims(t *testing.T) {
	assert := assert.New(t)

	uShape := "hexagon"
	// 1D data with more than one sample
	data := mat64.NewDense(2, 1, []float64{2, 3})
	dims, err := GridDims(data, uShape)
	assert.NoError(err)
	assert.EqualValues(dims, []int{1, 8})
	// 2D data with one sample
	data = mat64.NewDense(1, 2, []float64{2, 3})
	dims, err = GridDims(data, uShape)
	assert.NoError(err)
	assert.EqualValues(dims, []int{2, 2})
	// 2D+ data with more than one sample and hexagon uShape
	data = mat64.NewDense(6, 4, []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2,
		5.4, 3.9, 1.7, 0.4,
	})
	dims, err = GridDims(data, uShape)
	assert.NoError(err)
	assert.EqualValues(dims, []int{4, 3})
	// 2D+ data with more than one sample and rectangle uShape
	dims, err = GridDims(data, "rectangle")
	assert.NoError(err)
	assert.EqualValues(dims, []int{4, 3})
	// data matrix can't be nil
	dims, err = GridDims(nil, uShape)
	assert.Nil(dims)
	assert.Error(err)
}

func TestRandInit(t *testing.T) {
	assert := assert.New(t)

	min1, max1 := 1.2, 4.5
	min2, max2 := 3.4, 6.7
	data := []float64{min1, min2, max1, max2}
	inMx := mat64.NewDense(2, 2, data)
	randMx := mat64.NewDense(2, 2, nil)

	_, cols := inMx.Dims()
	// initialize random matrix
	err := RandInit(inMx, randMx)
	assert.NoError(err)
	for i := 0; i < cols; i++ {
		inCol := inMx.ColView(i)
		randCol := randMx.ColView(i)
		assert.True(mat64.Min(inCol) <= mat64.Min(randCol))
		assert.True(mat64.Max(inCol) >= mat64.Max(randCol))
	}

	// nil input matrix
	errString := "invalid data matrix: %v"
	err = RandInit(nil, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	// nil codebook matrix
	errString = "invalid codebook matrix: %v"
	err = RandInit(inMx, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	// negative number of rows
	errString = "data and codebook dimension mismatched: %d != %d"
	tmpMx := mat64.NewDense(2, 1, nil)
	err = RandInit(tmpMx, randMx)
	_, tCols := tmpMx.Dims()
	_, rCols := randMx.Dims()
	assert.EqualError(err, fmt.Sprintf(errString, tCols, rCols))
}

func TestLinInit(t *testing.T) {
	assert := assert.New(t)

	inMx := mat64.NewDense(6, 4, []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2,
		5.4, 3.9, 1.7, 0.4,
	})
	_, cols := inMx.Dims()

	xDim, yDim := 5, 2
	linMx, err := LinInit(inMx, []int{xDim, yDim})
	assert.NotNil(linMx)
	assert.NoError(err)
	// check if the dimensions are correct munits x datadim
	linR, linC := linMx.Dims()
	assert.Equal(linR, xDim*yDim)
	assert.Equal(linC, cols)
	// data is nil
	linMx, err = LinInit(nil, []int{1, 2})
	assert.Nil(linMx)
	assert.Error(err)
	// nil dimensions
	linMx, err = LinInit(inMx, nil)
	assert.Nil(linMx)
	assert.Error(err)
	// non positive dimensions supplied
	linMx, err = LinInit(inMx, []int{-1, 2})
	assert.Nil(linMx)
	assert.Error(err)
	// insufficient number of samples
	inMx = mat64.NewDense(1, 2, []float64{1, 1})
	linMx, err = LinInit(inMx, []int{5, 2})
	assert.Nil(linMx)
	assert.Error(err)
}

func TestGridCoords(t *testing.T) {
	assert := assert.New(t)

	// hexagon shape
	dims := []int{4, 2}
	mUnits := dims[0] * dims[1]
	mDims := len(dims)
	expMx := mat64.NewDense(mUnits, mDims, []float64{
		0.0, 0.0,
		0.5, 0.866,
		0.0, 1.732,
		0.5, 2.598,
		1.0, 0.0,
		1.5, 0.866,
		1.0, 1.732,
		1.5, 2.598})
	coords, err := GridCoords("hexagon", dims)
	assert.NotNil(coords)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(coords, expMx, 0.01))
	// rectangle shape
	dims = []int{3, 2}
	mUnits = dims[0] * dims[1]
	mDims = len(dims)
	expMx = mat64.NewDense(mUnits, mDims, []float64{
		0.0, 0.0,
		0.0, 1.0,
		0.0, 2.0,
		1.0, 0.0,
		1.0, 1.0,
		1.0, 2.0})
	coords, err = GridCoords("rectangle", dims)
	assert.NotNil(coords)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(coords, expMx, 0.01))
	// incorrect units shape
	coords, err = GridCoords("fooshape", []int{2, 2})
	assert.Nil(coords)
	assert.Error(err)
	// nil dimensions
	coords, err = GridCoords("hexagon", nil)
	assert.Nil(coords)
	assert.Error(err)
	// unsupported number of dimensions
	coords, err = GridCoords("hexagon", []int{1, 2, 3, 4})
	assert.Nil(coords)
	assert.Error(err)
	// negative plane dimensions
	coords, err = GridCoords("hexagon", []int{-1, 2})
	assert.Nil(coords)
	assert.Error(err)
}
