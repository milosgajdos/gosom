package utils

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInvariants(t *testing.T) {
	assert := assert.New(t)

	const (
		rows          = 20
		columns       = 2
		numOfClusters = 10
		min           = 0.0
		max           = 1.0
		vari          = 0.1
		randSeed      = 1
	)
	data := GenerateClusters(20, 2, 10, 0.0, 1.0, 0.1, 1)

	dRows, dCols := data.Dims()
	assert.Equal(rows, dRows)
	assert.Equal(columns, dCols)
	for i := 0; i < dRows; i++ {
		for j := 0; j < dCols; j++ {
			val := data.At(i, j)
			assert.True(val >= min-vari)
			assert.True(val <= max+vari)
		}
	}
}

func Test0Var(t *testing.T) {
	assert := assert.New(t)

	data := GenerateClusters(100, 2, 1, 0.0, 1.0, 0.0, 1)

	dRows, dCols := data.Dims()
	prevVal := make([]*float64, dCols)
	for i := 0; i < dRows; i++ {
		for j := 0; j < dCols; j++ {
			val := data.At(i, j)
			if prevVal[j] != nil {
				assert.Equal(*prevVal[j], val)
			}
			prevVal[j] = &val
		}
	}
}

func TestWithVar(t *testing.T) {
	assert := assert.New(t)

	const vari = 0.1
	data := GenerateClusters(100, 2, 1, 0.0, 1.0, vari, 1)

	dRows, dCols := data.Dims()
	prevVal := make([]*float64, dCols)
	for i := 0; i < dRows; i++ {
		for j := 0; j < dCols; j++ {
			val := data.At(i, j)
			if prevVal[j] != nil {
				assert.True(math.Abs(*prevVal[j]-val) < 2*vari)
			}
			prevVal[j] = &val
		}
	}
}
