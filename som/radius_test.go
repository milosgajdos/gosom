package som

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpRadius(t *testing.T) {
	testRadius(t, "exp")
}

func TestLinRadius(t *testing.T) {
	testRadius(t, "lin")
}

func TestDefaultRadius(t *testing.T) {
	testRadius(t, "some other")
}

func TestNegativeRadius(t *testing.T) {
	assert := assert.New(t)

	r, err := Radius(0, 1, "exp", -1.0)
	assert.True(math.IsNaN(r))
	assert.NotEmpty(err)
}

func testRadius(t *testing.T, strategy string) {
	assert := assert.New(t)

	radius0 := 100.0
	totalIterations := 100

	r, err := Radius(0, totalIterations, strategy, radius0)
	assert.Equal(radius0, r)
	assert.NoError(err)

	r, err = Radius(totalIterations-1, totalIterations, strategy, radius0)
	assert.InDelta(SmallestRadius, r, 0.01)
	assert.NoError(err)

}
