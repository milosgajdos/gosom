package som

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestRandInit(t *testing.T) {
	assert := assert.New(t)

	min1, max1 := 1.2, 4.5
	min2, max2 := 3.4, 6.7
	data := []float64{min1, min2, max1, max2}
	inMx := mat64.NewDense(2, 2, data)
	assert.NotNil(inMx)

	_, cols := inMx.Dims()
	rows := 4
	// initialize random matrix
	randMx, err := RandInit(inMx, rows, nil)
	assert.NotNil(randMx)
	assert.NoError(err)
	r, c := randMx.Dims()
	assert.Equal(rows, r)
	assert.Equal(cols, c)
	for i := 0; i < cols; i++ {
		inCol := inMx.ColView(i)
		randCol := randMx.ColView(i)
		assert.True(mat64.Min(inCol) <= mat64.Min(randCol))
		assert.True(mat64.Max(inCol) >= mat64.Max(randCol))
	}

	// nil input matrix
	randMx, err = RandInit(nil, rows, nil)
	assert.Nil(randMx)
	assert.Error(err)
	// negative number of rows
	randMx, err = RandInit(inMx, -9, nil)
	assert.Nil(randMx)
	assert.Error(err)
	// empty matrix
	emptyMx := mat64.NewDense(0, 0, nil)
	randMx, err = RandInit(emptyMx, 10, nil)
	assert.Nil(randMx)
	assert.Error(err)
}

func TestMakeXCoords(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		mUnits   int
		xDim     int
		expected []float64
	}{
		{6, 3, []float64{0.0, 0.0, 1.0, 1.0, 2.0, 2.0}},
		{6, 2, []float64{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}},
	}

	for _, tc := range testCases {
		xCoords := makeXCoords(tc.mUnits, tc.xDim)
		assert.EqualValues(xCoords, tc.expected)
	}
}

func TestMakeYCoords(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		mUnits   int
		yDim     int
		expected []float64
	}{

		{6, 2, []float64{0.0, 1.0, 0.0, 1.0, 0.0, 1.0}},
		{6, 3, []float64{0.0, 1.0, 2.0, 0.0, 1.0, 2.0}},
	}

	for _, tc := range testCases {
		yCoords := makeYCoords(tc.mUnits, tc.yDim)
		assert.EqualValues(yCoords, tc.expected)
	}
}

func TestPlaneGridCoords(t *testing.T) {
	assert := assert.New(t)

	// incorrect units shape
	coords, err := PlaneGridCoords("fooshape", []int{1, 2})
	assert.Nil(coords)
	assert.Error(err)
	// incprrect dimensions
	coords, err = PlaneGridCoords("hexagon", nil)
	assert.Nil(coords)
	assert.Error(err)
	coords, err = PlaneGridCoords("hexagon", []int{1, 2, 3})
	assert.Nil(coords)
	assert.Error(err)
	// incorrect number of units
	coords, err = PlaneGridCoords("hexagon", []int{-1, 2})
	assert.Nil(coords)
	assert.Error(err)
	// hexagon shape
	dims := []int{3, 2}
	mUnits := dims[0] * dims[1]
	mDims := len(dims)
	expMx := mat64.NewDense(mUnits, mDims, []float64{
		0.0, 0.0,
		0.5, 0.8660,
		1.0, 0.0,
		1.5, 0.8660,
		2.0, 0.0,
		2.5, 0.8660})
	coords, err = PlaneGridCoords("hexagon", dims)
	assert.NotNil(coords)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(coords, expMx, 0.01))
	// rectangle shape
	dims = []int{2, 3}
	mUnits = dims[0] * dims[1]
	mDims = len(dims)
	expMx = mat64.NewDense(mUnits, mDims, []float64{
		0.0, 0.0,
		0.0, 1.0,
		0.0, 2.0,
		1.0, 0.0,
		1.0, 1.0,
		1.0, 2.0})
	coords, err = PlaneGridCoords("rectangle", dims)
	assert.NotNil(coords)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(coords, expMx, 0.01))
}

func TestLinInit(t *testing.T) {
	assert := assert.New(t)

	//inMx := mat64.NewDense(3, 4, []float64{
	//	5.1, 3.5, 1.4, 0.2,
	//	4.9, 3.0, 1.4, 0.2,
	//	4.7, 3.2, 1.3, 0.2,
	//})
	//rows, cols := inMx.Dims()

	linMx, err := LinInit(nil, 10, nil)
	assert.Nil(linMx)
	assert.Error(err)
	// insufficient number of samples
	inMx := mat64.NewDense(1, 2, []float64{1, 1})
	linMx, err = LinInit(inMx, 10, nil)
	assert.Nil(linMx)
	assert.Error(err)
}
