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
