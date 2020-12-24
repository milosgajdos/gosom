package som

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/milosgajdos/gosom/pkg/utils"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	mSom   *MapConfig
	tSom   *TrainConfig
	dataMx *mat.Dense
)

func setup() {
	grid := &GridConfig{
		Size:   []int{2, 3},
		Type:   "planar",
		UShape: "hexagon",
	}

	cb := &CbConfig{
		InitFunc: RandInit,
	}

	// Init to default config
	mSom = &MapConfig{
		Grid: grid,
		Cb:   cb,
	}
	tSom = &TrainConfig{
		Algorithm: "seq",
		Radius:    10.0,
		RDecay:    "lin",
		NeighbFn:  Gaussian,
		LRate:     0.5,
		LDecay:    "lin",
	}
	// Create input data matrix
	data := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	dataMx = mat.NewDense(5, 4, data)
	// set the Codebook dimension to number of data columns
	mSom.Cb.Dim = 4
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// call with result of m.Run()
	os.Exit(retCode)
}

func mockInit(d *mat.Dense, dims []int) (*mat.Dense, error) {
	return nil, errors.New("Test error")
}

func TestNewMap(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// when nil init function, use RandInit
	origInitFunc := mSom.Cb.InitFunc
	mSom.Cb.InitFunc = nil
	m, err = NewMap(mSom, dataMx)
	assert.Nil(m)
	assert.Error(err)
	mSom.Cb.InitFunc = origInitFunc
	// incorrect init matrix
	m, err = NewMap(mSom, nil)
	assert.Nil(m)
	assert.Error(err)
	// init func that always returns error
	mSom.Cb.InitFunc = mockInit
	m, err = NewMap(mSom, dataMx)
	assert.Nil(m)
	assert.Error(err)
	mSom.Cb.InitFunc = RandInit
	// map dim must be positive number
	origDim := mSom.Cb.Dim
	mSom.Cb.Dim = -10
	m, err = NewMap(mSom, dataMx)
	assert.Nil(m)
	assert.Error(err)
	mSom.Cb.Dim = origDim
}

func TestCodebook(t *testing.T) {
	assert := assert.New(t)

	mapUnits := utils.IntProduct(mSom.Grid.Size)
	_, cols := dataMx.Dims()
	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	codebook := m.Codebook()
	assert.NotNil(codebook)
	cbRows, cbCols := codebook.Dims()
	assert.Equal(mapUnits, cbRows)
	assert.Equal(cols, cbCols)
}

func TestGrid(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	grid := m.Grid()
	assert.NotNil(grid)
	rows, cols := grid.Coords().Dims()
	assert.Equal(cols, len(mSom.Grid.Size))
	assert.Equal(rows, mSom.Grid.Size[0]*mSom.Grid.Size[1])
}

func TestUnitDist(t *testing.T) {
	assert := assert.New(t)

	mapUnits := utils.IntProduct(mSom.Grid.Size)
	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	unitDist, err := m.UnitDist()
	assert.NotNil(unitDist)
	assert.NoError(err)
	cbRows, cbCols := unitDist.Dims()
	assert.Equal(mapUnits, cbRows)
	assert.Equal(mapUnits, cbCols)
}

func TestMapBmus(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	bmus, err := m.BMUs(dataMx)
	assert.NoError(err)
	assert.NotNil(bmus)
	rows, _ := dataMx.Dims()
	assert.Equal(rows, len(bmus))
}

func TestTrain(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// incorrect number of iterations
	errString := "invalid number of iterations: %d"
	iters := -100
	err = m.Train(tSom, dataMx, iters)
	assert.EqualError(err, fmt.Sprintf(errString, iters))
	iters = 100
	// incorrect number of iterations
	errString = "invalid data supplied: %v"
	err = m.Train(tSom, nil, iters)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	// throw in incorrect training config
	origRadius := tSom.Radius
	tSom.Radius = -10
	err = m.Train(tSom, dataMx, iters)
	assert.Error(err)
	tSom.Radius = origRadius
	// default config should not throw any errrors
	err = m.Train(tSom, dataMx, iters)
	assert.NoError(err)
	// batch training with default settings
	origAlgorithm := tSom.Algorithm
	tSom.Algorithm = "batch"
	err = m.Train(tSom, dataMx, iters)
	assert.NoError(err)
	tSom.Algorithm = origAlgorithm
}

func TestMapQuantError(t *testing.T) {
	assert := assert.New(t)

	// default config should not throw any errors
	m, err := NewMap(mSom, dataMx)
	assert.NotNil(m)
	assert.NoError(err)
	// get quant error
	qe, err := m.QuantError(dataMx)
	assert.NoError(err)
	assert.True(qe > 0.0)
}
