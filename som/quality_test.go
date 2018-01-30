package som

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	qData = mat.NewDense(5, 4,
		[]float64{5.1, 3.5, 1.4, 0.1,
			4.9, 3.0, 1.4, 0.2,
			4.7, 3.2, 1.3, 0.3,
			4.6, 3.1, 1.5, 0.4,
			5.0, 3.6, 1.4, 0.5})
	qCbook = mat.NewDense(3, 4,
		[]float64{5.1, 3.5, 1.4, 0.1,
			4.9, 3.0, 1.4, 0.2,
			5.0, 3.6, 1.4, 0.5})
)

func TestQuantError(t *testing.T) {
	assert := assert.New(t)

	// nil data returns error
	errString := "invalid data supplied: %v"
	qe, err := QuantError(nil, qCbook)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(-1.0, qe)
	// nil codebook returns error
	errString = "invalid codebook supplied: %v"
	qe, err = QuantError(qData, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(-1.0, qe)
	// incorrect dimensions of codebook and data
	qCbookTmp := mat.NewDense(1, 3, []float64{5.1, 3.5, 1.4})
	qe, err = QuantError(qData, qCbookTmp)
	assert.Error(err)
	assert.Equal(-1.0, qe)
	// quant error must be bigger than 0.0
	qe, err = QuantError(qData, qCbook)
	assert.NoError(err)
	assert.True(qe >= 0.0)
}

func TestTopoProduct(t *testing.T) {
	assert := assert.New(t)

	// test grid
	qGrid, err := GridCoords("rectangle", []int{1, 3})
	assert.NoError(err)
	// nil codebook returns error
	errString := "invalid codebook supplied: %v"
	tp, err := TopoProduct(nil, qGrid)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(0.0, tp)
	// nil grid returns error
	errString = "invalid grid supplied: %v"
	tp, err = TopoProduct(qCbook, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(0.0, tp)
	// grid and codebook dimension mismatch
	qGridErr, err := GridCoords("rectangle", []int{3, 3})
	assert.NoError(err)
	errString = "Grid and codebook dimension mismatch"
	tp, err = TopoProduct(qCbook, qGridErr)
	assert.EqualError(err, errString)
	assert.Equal(0.0, tp)
	// this should go through without errors
	tp, err = TopoProduct(qCbook, qGrid)
	assert.NoError(err)
}

func TestTopoError(t *testing.T) {
	assert := assert.New(t)

	qGrid, err := GridCoords("rectangle", []int{1, 3})
	// nil data returns error
	errString := "invalid data supplied: %v"
	te, err := TopoError(nil, qCbook, qGrid)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(-1.0, te)
	// nil codebook returns error
	errString = "invalid codebook supplied: %v"
	te, err = TopoError(qData, nil, qGrid)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(-1.0, te)
	// nil grid returns error
	errString = "invalid grid supplied: %v"
	te, err = TopoError(qData, qCbook, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Equal(-1.0, te)
	// incorrect dimensions of codebook and data
	qCbookTmp := mat.NewDense(1, 3, []float64{5.1, 3.5, 1.4})
	te, err = TopoError(qData, qCbookTmp, qGrid)
	assert.Error(err)
	assert.Equal(-1.0, te)
	// correct run
	te, err = TopoError(qData, qCbook, qGrid)
	assert.NoError(err)
	assert.True(te > 0.0)
}
