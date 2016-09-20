package som

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestNewMap(t *testing.T) {
	assert := assert.New(t)
	// Init to default config
	c = &Config{
		Dims:     []int{2, 3},
		Grid:     "planar",
		UShape:   "hexagon",
		Radius:   0,
		RCool:    "lin",
		NeighbFn: "gaussian",
		LRate:    0,
		LCool:    "lin",
	}
	// Create input data matrix
	data := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	dataMx := mat64.NewDense(5, 4, data)

	// default config should not throw any errors
	m, err := NewMap(c, RandInit, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// incorrect config
	origLcool := c.LCool
	c.LCool = "foobar"
	m, err = NewMap(c, RandInit, dataMx)
	assert.Nil(m)
	assert.Error(err)
	c.LCool = origLcool
	// incorrect init function
	m, err = NewMap(c, nil, dataMx)
	assert.Nil(m)
	assert.Error(err)
}
