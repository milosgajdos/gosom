package som

import (
	"errors"
	"os"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/pkg/utils"
	"github.com/stretchr/testify/assert"
)

var (
	cSom   *Config
	dataMx *mat64.Dense
)

func setup() {
	// Init to default config
	cSom = &Config{
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
	dataMx = mat64.NewDense(5, 4, data)
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// call with result of m.Run()
	os.Exit(retCode)
}

func mockInit(d *mat64.Dense, dims []int) (*mat64.Dense, error) {
	return nil, errors.New("Test error")
}

func TestNewMap(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(cSom, RandInit, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// incorrect config
	origLcool := cSom.LCool
	cSom.LCool = "foobar"
	m, err = NewMap(cSom, RandInit, dataMx)
	assert.Nil(m)
	assert.Error(err)
	cSom.LCool = origLcool
	// when nil init function, use RandInit
	m, err = NewMap(cSom, nil, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// incorrect init matrix
	m, err = NewMap(cSom, RandInit, nil)
	assert.Nil(m)
	assert.Error(err)
	// incorrect number of map units
	origDims := cSom.Dims
	cSom.Dims = []int{0, 0}
	m, err = NewMap(cSom, RandInit, dataMx)
	assert.Nil(m)
	assert.Error(err)
	cSom.Dims = origDims
	// init func that always returns error
	m, err = NewMap(cSom, mockInit, dataMx)
	assert.Nil(m)
	assert.Error(err)
}

func TestCodebook(t *testing.T) {
	assert := assert.New(t)

	mapUnits := utils.IntProduct(c.Dims)
	_, cols := dataMx.Dims()
	// default config should not throw any errors
	m, err := NewMap(cSom, RandInit, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	codebook := m.Codebook()
	assert.NotNil(codebook)
	cbRows, cbCols := codebook.Dims()
	assert.Equal(mapUnits, cbRows)
	assert.Equal(cols, cbCols)
}
