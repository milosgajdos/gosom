package som

import (
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

func testLR(t *testing.T, strategy string) {
	assert := assert.New(t)

	learningRate0 := 100.0
	totalIterations := 100

	assert.Equal(learningRate0, LearningRate(0, totalIterations, strategy, learningRate0))
	assert.InDelta(SmallestRate, LearningRate(totalIterations-1, totalIterations, strategy, learningRate0), 0.01)
}
