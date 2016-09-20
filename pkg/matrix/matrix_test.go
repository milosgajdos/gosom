package matrix

import (
	"fmt"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

var (
	errInvMx = "Invalid matrix supplied: %v\n"
	errEmpMx = "Empty matrix supplied: %v\n"
)

func TestColsMax(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	colsMax := []float64{8.9, 10.0}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)
	// check cols
	max, err := ColsMax(mx)
	assert.NotNil(max)
	assert.NoError(err)
	assert.EqualValues(colsMax, max)
	// should get nil back
	mx = nil
	max, err = ColsMax(mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	// zero elements in matrix
	data = []float64{}
	mx = mat64.NewDense(0, 0, data)
	assert.NotNil(mx)
	max, err = ColsMax(mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errEmpMx, mx))
}

func TestColsMin(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	colsMin := []float64{1.2, 3.4}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)
	// check cols
	min, err := ColsMin(mx)
	assert.NotNil(min)
	assert.NoError(err)
	assert.EqualValues(colsMin, min)
	// should get nil back
	mx = nil
	min, err = ColsMin(mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	// zero elements in matrix
	data = []float64{}
	mx = mat64.NewDense(0, 0, data)
	assert.NotNil(mx)
	min, err = ColsMin(mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errEmpMx, mx))
}

func TestMakeRandom(t *testing.T) {
	assert := assert.New(t)

	// create new matrix
	rows, cols := 2, 3
	min, max := 1.0, 2.0
	randMx, err := MakeRandom(rows, cols, min, max)
	assert.NotNil(randMx)
	assert.NoError(err)
	r, c := randMx.Dims()
	assert.Equal(r, rows)
	assert.Equal(c, cols)
	for i := 0; i < c; i++ {
		col := randMx.ColView(i)
		assert.True(max >= mat64.Max(col))
		assert.True(min <= mat64.Min(col))
	}
	// Can't create new matrix
	randMx, err = MakeRandom(rows, -6, min, max)
	assert.Nil(randMx)
	assert.Error(err)
	// Can't create new matrix
	randMx, err = MakeRandom(-10, cols, min, max)
	assert.Nil(randMx)
	assert.Error(err)
}

func TestConstant(t *testing.T) {
	assert := assert.New(t)

	// all elements must be equal to 1.0
	constVec := []float64{1.0, 1.0, 1.0, 1.0}
	constMx := mat64.NewDense(2, 2, constVec)
	mx, err := MakeConstant(2, 2, 1.0)
	assert.NotNil(mx)
	assert.True(mat64.Equal(constMx, mx))
	// Can't create new matrix
	constMx, err = MakeConstant(3, -6, 1.0)
	assert.Nil(constMx)
	assert.Error(err)
	// Can't create new matrix
	constMx, err = MakeConstant(-3, 10, 1.0)
	assert.Nil(constMx)
	assert.Error(err)
}
