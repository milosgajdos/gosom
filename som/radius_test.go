package som

import (
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

func testRadius(t *testing.T, strategy string) {
	assert := assert.New(t)

	radius0 := 100.0
	totalIterations := 100

	assert.Equal(radius0, Radius(0, totalIterations, strategy, radius0))
	assert.InDelta(1.0, Radius(totalIterations, totalIterations, strategy, radius0), 0.01)

}
