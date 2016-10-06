package som

import (
	"testing"

	"math"
	"math/rand"

	"github.com/stretchr/testify/assert"
)

func TestGaussian(t *testing.T) {
	assert.Equal(t, 1.0, Gaussian(0.0, rand.Float64()))

	assert.Equal(t, 0.0, Gaussian(math.Inf(1), rand.Float64()))

	assert.InDelta(t, 1/math.E, Gaussian(1.0, 1.0/math.Sqrt2), 0.01)
}
