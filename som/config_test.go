package som

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func makeDefaultMapCfg() *MapConfig {
	grid := &GridConfig{
		Size:   []int{2, 3},
		Type:   "planar",
		UShape: "hexagon",
	}

	cb := &CbConfig{
		Dim:      5,
		InitFunc: RandInit,
	}

	return &MapConfig{
		Grid: grid,
		Cb:   cb,
	}
}

func makeDefaultTrainConfig() *TrainConfig {
	return &TrainConfig{
		Algorithm: "seq",
		Radius:    10.0,
		RDecay:    "lin",
		NeighbFn:  "gaussian",
		LRate:     0.5,
		LDecay:    "lin",
	}
}

func TestValidateGridSize(t *testing.T) {
	assert := assert.New(t)

	mc := makeDefaultMapCfg()
	errDimLen := "unsupported number of SOM grid dimensions supplied: %d"
	errDimVal := "incorrect SOM grid dimensions supplied: %v"
	wrongDims := []int{-1, 2}
	singDims := []int{1, 1}
	testCases := []struct {
		size   []int
		expErr bool
		errStr string
	}{
		{[]int{1}, true, fmt.Sprintf(errDimLen, 1)},
		{[]int{}, true, fmt.Sprintf(errDimLen, 0)},
		{[]int{1, 2}, false, ""},
		{singDims, true, fmt.Sprintf(errDimVal, singDims)},
		{wrongDims, true, fmt.Sprintf(errDimVal, wrongDims)},
	}

	size := mc.Grid.Size
	for _, tc := range testCases {
		mc.Grid.Size = tc.size
		err := validateGridConfig(mc.Grid)
		if tc.expErr {
			assert.EqualError(err, tc.errStr)
		} else {
			assert.NoError(err)
		}
	}
	mc.Grid.Size = size
}

func TestValidateGridType(t *testing.T) {
	assert := assert.New(t)

	mc := makeDefaultMapCfg()
	errString := "unsupported SOM grid type: %s"
	testCases := []struct {
		grid   string
		expErr bool
	}{
		{"planar", false},
		{"foobar", true},
	}

	grid := mc.Grid.Type
	for _, tc := range testCases {
		mc.Grid.Type = tc.grid
		err := validateGridConfig(mc.Grid)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, mc.Grid.Type))
		} else {
			assert.NoError(err)
		}
	}
	mc.Grid.Type = grid
}

func TestValidateGridUshape(t *testing.T) {
	assert := assert.New(t)

	mc := makeDefaultMapCfg()
	errString := "unsupported SOM unit shape: %s"
	testCases := []struct {
		ushape string
		expErr bool
	}{
		{"hexagon", false},
		{"foobar", true},
		{"rectangle", false},
	}

	uShape := mc.Grid.UShape
	for _, tc := range testCases {
		mc.Grid.UShape = tc.ushape
		err := validateGridConfig(mc.Grid)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, mc.Grid.UShape))
		} else {
			assert.NoError(err)
		}
	}
	mc.Grid.UShape = uShape
}

func TestValidateCbInitFunc(t *testing.T) {
	assert := assert.New(t)

	mc := makeDefaultMapCfg()
	errString := "invalid InitFunc: %v"
	testCases := []struct {
		initFunc CodebookInitFunc
		expErr   bool
	}{
		{RandInit, false},
		{nil, true},
		{LinInit, false},
	}

	initFunc := mc.Cb.InitFunc
	for _, tc := range testCases {
		mc.Cb.InitFunc = tc.initFunc
		err := validateCbConfig(mc.Cb)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, mc.Cb.InitFunc))
		} else {
			assert.NoError(err)
		}
	}
	mc.Cb.InitFunc = initFunc
}

func TestValidateAlgorithm(t *testing.T) {
	assert := assert.New(t)

	tr := makeDefaultTrainConfig()
	errString := "invalid SOM training algorithm: %s"
	testCases := []struct {
		method string
		expErr bool
	}{
		{"seq", false},
		{"foobar", true},
		{"batch", false},
	}

	origAlgorithm := tr.Algorithm
	for _, tc := range testCases {
		tr.Algorithm = tc.method
		err := validateTrainConfig(tr)
		if tc.expErr {
			assert.EqualError(err, fmt.Sprintf(errString, tr.Algorithm))
		} else {
			assert.NoError(err)
		}
	}
	tr.Algorithm = origAlgorithm
}

func TestValidateRadius(t *testing.T) {
	assert := assert.New(t)

	tr := makeDefaultTrainConfig()
	errString := "invalid SOM unit radius: %f"
	testCases := []struct {
		radius float64
		expErr bool
	}{
		{1.0, false},
		{-10.0, true},
		{0.0, false},
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

	tr := makeDefaultTrainConfig()
	errString := "unsupported Radius decay strategy: %s"
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

	tr := makeDefaultTrainConfig()
	errString := "unsupported Neighbourhood function: %s"
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

	tr := makeDefaultTrainConfig()
	errString := "invalid SOM learning rate: %f"
	testCases := []struct {
		lrate  float64
		expErr bool
	}{
		{1.0, false},
		{-10.0, true},
		{0.0, false},
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

	tr := makeDefaultTrainConfig()
	errString := "unsupported Learning rate decay strategy: %s"
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
