package som

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpRadius(t *testing.T) {
	assert := assert.New(t)

	radius0 := 100.0
	totalIterations := 100
	strategy := "exp"

	assert.Equal(radius0, Radius(0, totalIterations, strategy, radius0))
	assert.InDelta(1.0, Radius(totalIterations, totalIterations, strategy, radius0), 0.01)
}

func TestLinRadius(t *testing.T) {
	assert := assert.New(t)

	radius0 := 100.0
	totalIterations := 100
	strategy := "lin"

	assert.Equal(radius0, Radius(0, totalIterations, strategy, radius0))
	assert.InDelta(1.0, Radius(totalIterations, totalIterations, strategy, radius0), 0.01)
}
