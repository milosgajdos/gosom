package matrix

import (
	"fmt"
	"testing"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

var (
	errInvMx     = "Invalid matrix supplied: %v\n"
	errInvColsMx = "Invalid number of columns supplied: %v\n"
	errInvRowsMx = "Invalid number of rows supplied: %v\n"
	errExcCols   = "Column count exceeds matrix dimensions: %d\n"
	errExcRows   = "Row count exceeds matrix dimensions: %d\n"
)

func TestRowsColsMax(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	colsMax := []float64{8.9, 10.0}
	rowsMax := []float64{3.4, 6.7, 10.0}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)

	rows, cols := mx.Dims()
	// check cols max
	max, err := ColsMax(cols, mx)
	assert.NotNil(max)
	assert.NoError(err)
	assert.EqualValues(colsMax, max)
	// check rows max
	max, err = RowsMax(rows, mx)
	assert.NotNil(max)
	assert.NoError(err)
	assert.EqualValues(rowsMax, max)
	// requested number of cols exceeds matrix dims
	max, err = ColsMax(cols+1, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errExcCols, cols+1))
	// requested number of rows exceeds matrix dims
	max, err = RowsMax(rows+1, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errExcRows, rows+1))
	// should get nil back
	mx = nil
	max, err = ColsMax(cols, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	max, err = RowsMax(rows, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	// zero elements in matrix
	data = []float64{}
	mx = mat64.NewDense(0, 0, data)
	assert.NotNil(mx)
	max, err = ColsMax(cols, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errInvColsMx, mx))
	max, err = RowsMax(rows, mx)
	assert.Nil(max)
	assert.EqualError(err, fmt.Sprintf(errInvRowsMx, mx))
}

func TestColsMin(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	colsMin := []float64{1.2, 3.4}
	rowsMin := []float64{1.2, 4.5, 8.9}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)

	rows, cols := mx.Dims()
	// check cols
	min, err := ColsMin(cols, mx)
	assert.NotNil(min)
	assert.NoError(err)
	assert.EqualValues(colsMin, min)
	// check rows
	min, err = RowsMin(rows, mx)
	assert.NotNil(min)
	assert.NoError(err)
	assert.EqualValues(rowsMin, min)
	// requested number of cols exceeds matrix dims
	min, err = ColsMax(cols+1, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errExcCols, cols+1))
	// requested number of rows exceeds matrix dims
	min, err = RowsMin(rows+1, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errExcRows, rows+1))
	// should get nil back
	mx = nil
	min, err = ColsMin(cols, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	min, err = RowsMin(rows, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errInvMx, mx))
	// zero elements in matrix
	data = []float64{}
	mx = mat64.NewDense(0, 0, data)
	assert.NotNil(mx)
	min, err = ColsMin(cols, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errInvColsMx, mx))
	min, err = RowsMin(rows, mx)
	assert.Nil(min)
	assert.EqualError(err, fmt.Sprintf(errInvRowsMx, mx))
}

func TestColsMean(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)
	colsMean := []float64{4.8667, 6.7000}

	_, cols := mx.Dims()
	// check cols
	me, err := ColsMean(cols, mx)
	assert.NotNil(me)
	assert.NoError(err)
	assert.True(floats.EqualApprox(colsMean, me, 0.01))
}

func TestColsStdev(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)
	colsStdev := []float64{3.8631, 3.3000}

	_, cols := mx.Dims()
	// check cols
	sd, err := ColsStdev(cols, mx)
	assert.NotNil(sd)
	assert.NoError(err)
	assert.True(floats.EqualApprox(colsStdev, sd, 0.01))
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

func TestMakeConstant(t *testing.T) {
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

func TestAddConstant(t *testing.T) {
	assert := assert.New(t)

	// all elements must be equal to 1.0
	val := 0.5
	mx := mat64.NewDense(2, 2, []float64{1.0, 2.0, 2.5, 2.5})
	mc := mat64.NewDense(2, 2, []float64{1.5, 2.5, 3.0, 3.0})

	mx, err := AddConst(val, mx)
	assert.NotNil(mx)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(mx, mc, 0.01))
	// incorrect matrix passed in
	mx, err = AddConst(val, nil)
	assert.Nil(mx)
	assert.Error(err)
}
