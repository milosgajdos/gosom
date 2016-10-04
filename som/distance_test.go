package som

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestDistanceMatrix(t *testing.T) {
	assert := assert.New(t)

	one := mat64.NewDense(2, 2, []float64{
		0.0, 0.0,
		1.0, 0.0,
	})

	oneR, _ := one.Dims()
	oneOutExpected := mat64.NewDense(oneR, oneR, []float64{
		0.0, 1.0,
		1.0, 0.0,
	})

	oneOut := DistanceMatrix("euclidean", one)

	assert.True(mat64.EqualApprox(oneOutExpected, oneOut, 0.01))

}
