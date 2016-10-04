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

	zero := mat64.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		33.0, 33.0, 33.0,
	})

	zeroR, _ := zero.Dims()
	zeroOutExpected := mat64.NewDense(zeroR, zeroR, []float64{
		0.0, 0.0,
		0.0, 0.0,
	})

	zeroOut := DistanceMatrix("euclidean", zero)

	assert.True(mat64.EqualApprox(zeroOutExpected, zeroOut, 0.01))

	negative := mat64.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		133.0, 33.0, 33.0,
	})

	negativeR, _ := negative.Dims()
	negativeOutExpected := mat64.NewDense(negativeR, negativeR, []float64{
		0.0, 100.0,
		100.0, 0.0,
	})

	negativeOut := DistanceMatrix("euclidean", negative)

	assert.True(mat64.EqualApprox(negativeOutExpected, negativeOut, 0.01))

}
