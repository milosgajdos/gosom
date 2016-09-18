package som

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	// Init to default config
	c = &Config{
		Init:     "random",
		Dims:     []int{2, 3},
		Grid:     "planar",
		UShape:   "hexagon",
		Radius:   0,
		RCool:    "lin",
		NeighbFn: "gaussian",
		LRate:    0,
		LCool:    "lin",
	}
)

func TestValidateInit(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported SOM initialization type: %s\n"
	testCases := []struct {
		init   string
		expErr bool
	}{
		{"random", false},
		{"linear", false},
		{"foobar", true},
	}

	origInit := c.Init
	for _, tc := range testCases {
		c.Init = tc.init
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.Init))
		} else {
			assert.NoError(err)
		}
	}
	c.Init = origInit
}

func TestValidateDims(t *testing.T) {
	assert := assert.New(t)

	errDimLen := "Incorrect number of dimensions supplied: %d\n"
	errDimVal := "Incorrect SOM dimensions supplied: %v\n"
	wrongDims := []int{-1, 2}
	testCases := []struct {
		dims   []int
		expErr bool
		errStr string
	}{
		{[]int{1}, true, fmt.Sprintf(errDimLen, 1)},
		{[]int{}, true, fmt.Sprintf(errDimLen, 0)},
		{[]int{1, 2}, false, ""},
		{wrongDims, true, fmt.Sprintf(errDimVal, wrongDims)},
	}

	origDims := c.Dims
	for _, tc := range testCases {
		c.Dims = tc.dims
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, tc.errStr)
		} else {
			assert.NoError(err)
		}
	}
	c.Dims = origDims
}

func TestValidateGrid(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported SOM grid type: %s\n"
	testCases := []struct {
		grid   string
		expErr bool
	}{
		{"planar", false},
		{"foobar", true},
	}

	origGrid := c.Grid
	for _, tc := range testCases {
		c.Grid = tc.grid
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.Grid))
		} else {
			assert.NoError(err)
		}
	}
	c.Grid = origGrid
}

func TestValidateUshape(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported SOM unit shape: %s\n"
	testCases := []struct {
		ushape string
		expErr bool
	}{
		{"hexagon", false},
		{"foobar", true},
		{"rectangle", false},
	}

	origUShape := c.UShape
	for _, tc := range testCases {
		c.UShape = tc.ushape
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.UShape))
		} else {
			assert.NoError(err)
		}
	}
	c.UShape = origUShape
}

func TestValidateRadius(t *testing.T) {
	assert := assert.New(t)

	errString := "Invalid SOM unit radius: %d\n"
	testCases := []struct {
		radius int
		expErr bool
	}{
		{1, false},
		{-10, true},
		{0, false},
	}

	origRadius := c.Radius
	for _, tc := range testCases {
		c.Radius = tc.radius
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.Radius))
		} else {
			assert.NoError(err)
		}
	}
	c.Radius = origRadius
}

func TestValidateRCool(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported Radius cooling strategy: %s\n"
	testCases := []struct {
		rcool  string
		expErr bool
	}{
		{"lin", false},
		{"foobar", true},
		{"exp", false},
		{"inv", false},
	}

	origRCool := c.RCool
	for _, tc := range testCases {
		c.RCool = tc.rcool
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.RCool))
		} else {
			assert.NoError(err)
		}
	}
	c.RCool = origRCool
}

func TestValidateNeighbFn(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported Neighbourhood function: %s\n"
	testCases := []struct {
		neighbFn string
		expErr   bool
	}{
		{"gaussian", false},
		{"foobar", true},
		{"mexican", false},
		{"bubble", false},
	}

	origNeighbFn := c.NeighbFn
	for _, tc := range testCases {
		c.NeighbFn = tc.neighbFn
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.NeighbFn))
		} else {
			assert.NoError(err)
		}
	}
	c.NeighbFn = origNeighbFn
}

func TestValidateLRate(t *testing.T) {
	assert := assert.New(t)

	errString := "Invalid SOM learning rate: %d\n"
	testCases := []struct {
		lrate  int
		expErr bool
	}{
		{1, false},
		{-10, true},
		{0, false},
	}

	origLRate := c.LRate
	for _, tc := range testCases {
		c.LRate = tc.lrate
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.LRate))
		} else {
			assert.NoError(err)
		}
	}
	c.LRate = origLRate
}

func TestValidateLCool(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported Learning rate cooling strategy: %s\n"
	testCases := []struct {
		lcool  string
		expErr bool
	}{
		{"lin", false},
		{"exp", false},
		{"foobar", true},
		{"inv", false},
	}

	origLCool := c.LCool
	for _, tc := range testCases {
		c.LCool = tc.lcool
		err := validateConfig(c)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, c.LCool))
		} else {
			assert.NoError(err)
		}
	}
	c.LCool = origLCool
}
