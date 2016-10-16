package som

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/pkg/utils"
)

// CodebookInitFunc defines SOM codebook initialization function
type CodebookInitFunc func(*mat64.Dense, []int) (*mat64.Dense, error)

// CoordsInitFunc defines SOM grid coordinates initialization function
type CoordsInitFunc func(string, []int) (*mat64.Dense, error)

// NeighbFunc defines SOM neighbourhood function
type NeighbFunc func(float64, float64) float64

// Map is a Self Organizing Map (SOM)
type Map struct {
	// codebook is a matrix which contains SOM codebook vectors
	// codebook dimensions: SOM units x data features
	codebook *mat64.Dense
	// gridDist is a symmetric hollow matrix that maps distances between SOM units
	// gridDist dimesions: SOM units x SOM units
	gridDist *mat64.Dense
	// bmus stores codebook row indices of Best Match Units (BMU) for each data sample
	// bmus length is equal to the number of the input data samples
	bmus []int
}

// NewMap creates new SOM based on the provided configuration and input data
// NewMap allows you to pass in SOM codebook init function that is used to initialize
// SOM codebook vectors to initial values. If codebook InitFunc is nil, random initialization
// is used. NewMap returns error if the provided configuration is not valid or if the data matrix
// is nil or if the codebook matrix could not be initialized.
func NewMap(c *MapConfig, data *mat64.Dense) (*Map, error) {
	// validate the map configuration
	if err := validateMapConfig(c); err != nil {
		return nil, err
	}
	// if nil codebook init function is passed in, use random init
	if c.InitFunc == nil {
		c.InitFunc = RandInit
	}
	// if input data is empty throw error
	if data == nil {
		return nil, fmt.Errorf("Invalid input data: %v\n", data)
	}
	// compute the number of map units
	mUnits := utils.IntProduct(c.Dims)
	if mUnits <= 1 {
		return nil, fmt.Errorf("Incorrect map size dimensions: %v\n", c.Dims)
	}
	// initialize codebook
	codebook, err := c.InitFunc(data, c.Dims)
	if err != nil {
		return nil, err
	}
	// grid coordinates matrix
	gridCoords, err := GridCoords(c.UShape, c.Dims)
	if err != nil {
		return nil, err
	}
	// grid distance matrix
	gridDist, err := DistanceMx("euclidean", gridCoords)
	if err != nil {
		return nil, err
	}
	rows, _ := data.Dims()
	bmus := make([]int, rows)
	// return pointer to new map
	return &Map{
		codebook: codebook,
		gridDist: gridDist,
		bmus:     bmus,
	}, nil
}

// Codebook returns a matrix which contains SOM codebook vectors
func (m Map) Codebook() *mat64.Dense {
	return m.codebook
}

// GridDist returns a matrix which contains Euclidean distances between SOM units
func (m Map) GridDist() *mat64.Dense {
	return m.gridDist
}

// BMUs returns a slice which contains indices of Best Match Units (BMUs) of each input vector
func (m Map) BMUs() []int {
	return m.bmus
}

// Train runs a SOM training for a given data set and training configuration parameters
// Train modifies the SOM codebook vectors according to the chosen training method.
// It returns error if the supplied training configuration is invalid
func (m *Map) Train(tc *TrainConfig, data *mat64.Dense, iters int) error {
	// number of iterations must be a positive integer
	if iters <= 0 {
		return fmt.Errorf("Invalid number of iterations: %d\n", iters)
	}
	// validate the training configuration
	if err := validateTrainConfig(tc); err != nil {
		return err
	}
	return nil
}
