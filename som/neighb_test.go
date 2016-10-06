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

func TestMexican(t *testing.T) {
	radius := 5.0

	atRadius := Mexican(radius, radius)

	// radius * (1.0 +- 1.0/100.0) is there to be 100% sure we never land on the radius if rand.Float64()==0
	insideRadius := make([]float64, 100)
	for i := 0; i < len(insideRadius); i++ {
		insideRadius[i] = Mexican((rand.Float64()-0.5)*2*radius*(1.0-1.0/100.0), radius)
	}

	outsideRadius := make([]float64, 100)
	for i := 0; i < len(outsideRadius); i++ {
		point := rand.Float64()*100.00 + radius*(1.0+1.0/100.0)
		if i%2 == 0 {
			point = -point
		}
		outsideRadius[i] = Mexican(point, radius)
	}

	assert.InDelta(t, 0.0, atRadius, 0.01)
	for i := 0; i < len(insideRadius); i++ {
		assert.True(t, insideRadius[i] > 0)
	}
	for i := 0; i < len(outsideRadius); i++ {
		assert.True(t, outsideRadius[i] < 0)
	}

}

func TestBubble(t *testing.T) {
	radius := 1.0
	diff := radius / 10.0

	assert.Equal(t, 1.0, Bubble(radius-diff, radius))
	assert.Equal(t, 0.0, Bubble(radius+diff, radius))
	assert.Equal(t, 1.0, Bubble(radius, radius))
}
