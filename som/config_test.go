package som

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	// Init map to default config
	mc = &MapConfig{
		Dims:   []int{2, 3},
		Grid:   "planar",
		UShape: "hexagon",
	}
	// default training configuration
	tr = &TrainConfig{
		Method:   "seq",
		Radius:   0,
		RDecay:   "lin",
		NeighbFn: "gaussian",
		LRate:    0,
		LDecay:   "lin",
	}
)

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

	origDims := mc.Dims
	for _, tc := range testCases {
		mc.Dims = tc.dims
		err := validateMapConfig(mc)
		if tc.expErr {
			assert.EqualError(err, tc.errStr)
		} else {
			assert.NoError(err)
		}
	}
	mc.Dims = origDims
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

	origGrid := mc.Grid
	for _, tc := range testCases {
		mc.Grid = tc.grid
		err := validateMapConfig(mc)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, mc.Grid))
		} else {
			assert.NoError(err)
		}
	}
	mc.Grid = origGrid
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

	origUShape := mc.UShape
	for _, tc := range testCases {
		mc.UShape = tc.ushape
		err := validateMapConfig(mc)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, mc.UShape))
		} else {
			assert.NoError(err)
		}
	}
	mc.UShape = origUShape
}

func TestValidateMethod(t *testing.T) {
	assert := assert.New(t)

	errString := "Invalid SOM training method: %s\n"
	testCases := []struct {
		method string
		expErr bool
	}{
		{"seq", false},
		{"foobar", true},
		{"batch", false},
	}

	origMethod := tr.Method
	for _, tc := range testCases {
		tr.Method = tc.method
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.Method))
		} else {
			assert.NoError(err)
		}
	}
	tr.Method = origMethod
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

	origRadius := tr.Radius
	for _, tc := range testCases {
		tr.Radius = tc.radius
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.Radius))
		} else {
			assert.NoError(err)
		}
	}
	tr.Radius = origRadius
}

func TestValidateRDecay(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported Radius decay strategy: %s\n"
	testCases := []struct {
		decay  string
		expErr bool
	}{
		{"lin", false},
		{"foobar", true},
		{"exp", false},
		{"inv", false},
	}

	origRDecay := tr.RDecay
	for _, tc := range testCases {
		tr.RDecay = tc.decay
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.RDecay))
		} else {
			assert.NoError(err)
		}
	}
	tr.RDecay = origRDecay
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

	origNeighbFn := tr.NeighbFn
	for _, tc := range testCases {
		tr.NeighbFn = tc.neighbFn
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.NeighbFn))
		} else {
			assert.NoError(err)
		}
	}
	tr.NeighbFn = origNeighbFn
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

	origLRate := tr.LRate
	for _, tc := range testCases {
		tr.LRate = tc.lrate
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.LRate))
		} else {
			assert.NoError(err)
		}
	}
	tr.LRate = origLRate
}

func TestValidateLDecay(t *testing.T) {
	assert := assert.New(t)

	errString := "Unsupported Learning rate decay strategy: %s\n"
	testCases := []struct {
		decay  string
		expErr bool
	}{
		{"lin", false},
		{"exp", false},
		{"foobar", true},
		{"inv", false},
	}

	origLDecay := tr.LDecay
	for _, tc := range testCases {
		tr.LDecay = tc.decay
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.LDecay))
		} else {
			assert.NoError(err)
		}
	}
	tr.LDecay = origLDecay
}
