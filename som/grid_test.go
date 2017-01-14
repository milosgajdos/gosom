package som

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestNewGrid(t *testing.T) {
	assert := assert.New(t)

	gCfg := &GridConfig{
		Size:   []int{2, 3},
		Type:   "planar",
		UShape: "hexagon",
	}
	// returns grid
	g, err := NewGrid(gCfg)
	assert.NotNil(g)
	assert.NoError(err)
	// test dims
	gSize := g.Size()
	assert.EqualValues(gCfg.Size, gSize)
	// test UShape
	uShape := g.UShape()
	assert.Equal(gCfg.UShape, uShape)
	// test coords
	coords := g.Coords()
	rows, cols := coords.Dims()
	assert.Equal(cols, len(gCfg.Size))
	assert.Equal(rows, gCfg.Size[0]*gCfg.Size[1])
	// test error cases
	origDims := gCfg.Size
	gCfg.Size = []int{1}
	g, err = NewGrid(gCfg)
	assert.Nil(g)
	assert.Error(err)
	// negative dimensions
	gCfg.Size = []int{-2, 2}
	g, err = NewGrid(gCfg)
	assert.Nil(g)
	assert.Error(err)
	// unit dimension given
	gCfg.Size = []int{-2, 2}
	g, err = NewGrid(gCfg)
	assert.Nil(g)
	assert.Error(err)
	gCfg.Size = origDims
}

func TestGridSize(t *testing.T) {
	assert := assert.New(t)

	uShape := "hexagon"
	// 1D data with more than one sample
	data := mat64.NewDense(2, 1, []float64{2, 3})
	dims, err := GridSize(data, uShape)
	assert.NoError(err)
	assert.EqualValues(dims, []int{1, 8})
	// 2D data with one sample
	data = mat64.NewDense(1, 2, []float64{2, 3})
	dims, err = GridSize(data, uShape)
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
	dims, err = GridSize(data, uShape)
	assert.NoError(err)
	assert.EqualValues(dims, []int{4, 3})
	// 2D+ data with more than one sample and rectangle uShape
	dims, err = GridSize(data, "rectangle")
	assert.NoError(err)
	assert.EqualValues(dims, []int{4, 3})
	// data matrix can't be nil
	dims, err = GridSize(nil, uShape)
	assert.Nil(dims)
	assert.Error(err)
}

func TestRandInit(t *testing.T) {
	assert := assert.New(t)

	min1, max1 := 1.2, 4.5
	min2, max2 := 3.4, 6.7
	data := []float64{min1, min2, max1, max2}
	inMx := mat64.NewDense(2, 2, data)
	assert.NotNil(inMx)

	_, cols := inMx.Dims()
	// initialize random matrix
	xDim, yDim := 2, 2
	randMx, err := RandInit(inMx, []int{xDim, yDim})
	assert.NotNil(randMx)
	assert.NoError(err)
	r, c := randMx.Dims()
	assert.Equal(xDim*yDim, r)
	assert.Equal(cols, c)
	for i := 0; i < cols; i++ {
		inCol := inMx.ColView(i)
		randCol := randMx.ColView(i)
		assert.True(mat64.Min(inCol) <= mat64.Min(randCol))
		assert.True(mat64.Max(inCol) >= mat64.Max(randCol))
	}

	// nil input matrix
	randMx, err = RandInit(nil, nil)
	assert.Nil(randMx)
	assert.Error(err)
	// nil dimensions
	randMx, err = RandInit(inMx, nil)
	assert.Nil(randMx)
	assert.Error(err)
	// negative number of rows
	randMx, err = RandInit(inMx, []int{-4, 3})
	assert.Nil(randMx)
	assert.Error(err)
	// empty matrix
	emptyMx := mat64.NewDense(0, 0, nil)
	randMx, err = RandInit(emptyMx, []int{2, 3})
	assert.Nil(randMx)
	assert.Error(err)
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
