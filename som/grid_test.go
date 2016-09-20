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
	randMx, err := RandInit(inMx, rows)
	assert.NotNil(randMx)
	assert.NoError(err)
	r, c := randMx.Dims()
	assert.Equal(rows, r)
	assert.Equal(cols, c)
	for i := 0; i < c; i++ {
		inCol := inMx.ColView(i)
		randCol := randMx.ColView(i)
		assert.True(inCol.At(0, 0) <= mat64.Min(randCol))
		assert.True(inCol.At(1, 0) >= mat64.Max(randCol))
	}

	// nil input matrix
	randMx, err = RandInit(nil, rows)
	assert.Nil(randMx)
	assert.Error(err)
	// negative number of rows
	randMx, err = RandInit(inMx, -9)
	assert.Nil(randMx)
	assert.Error(err)
	// empty matrix
	emptyMx := mat64.NewDense(0, 0, nil)
	randMx, err = RandInit(emptyMx, 10)
	assert.Nil(randMx)
	assert.Error(err)
}
