package som

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpLR(t *testing.T) {
	testLR(t, "exp")
}

func TestLinLR(t *testing.T) {
	testLR(t, "lin")
}

func TestDefaultLR(t *testing.T) {
	testLR(t, "some other")
}

func TestNegativeLR(t *testing.T) {
	assert := assert.New(t)

	r, err := LRate(0, 1, "exp", -1.0)
	assert.True(math.IsNaN(r))
	assert.NotEmpty(err)
}

func testLR(t *testing.T, strategy string) {
	assert := assert.New(t)

	learningRate0 := 100.0
	totalIterations := 100

	lr, err := LRate(0, totalIterations, strategy, learningRate0)
	assert.Equal(learningRate0, lr)
	assert.NoError(err)

	lr, err = LRate(totalIterations-1, totalIterations, strategy, learningRate0)
	assert.InDelta(MinLRate, lr, 0.01)
	assert.NoError(err)
}
