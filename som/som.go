package som

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// CodebookInitFunc defines SOM codebook initialization function
type CodebookInitFunc func(*mat64.Dense, int) (*mat64.Dense, error)

// CoordsInitFunc defines SOM grid coordinates initialization function
type CoordsInitFunc func(string) (*mat64.Dense, error)

// NeighbFunc defines SOM neighbourhood function
type NeighbFunc func(float64, float64) float64

// Map is a Self Organizing Map (SOM)
type Map struct {
	// codebook is a matrix which contains SOM codebook vectors
	// codebook dimensions: SOM units x data features
	codebook *mat64.Dense
	// uDist is a matrix that maps distances between SOM units
	// uDist dimesions: SOM units x SOM units
	uDist *mat64.Dense
	// uCoords is a matrix that holds SOM units coordinates
	// uCoords: SOM units x 2
	uCoords *mat64.Dense
	// bmus stores codebook row indices of Best Match Units (BMU) for each data sample
	// bmus length is equal to the number of the input data samples
	bmus []int
}

// NewMap creates new SOM based on the provided configuration and input data stored passed in as a
// parameter. NewMap allows you to pass in SOM codebook init function that is used to initialize
// SOM codebook vectors to initial values.
// It returns error if the provided configuration is not valid or data matrix passed in is empty
// or if the codebook init function passed in is nil.
func NewMap(c *Config, initFunc CodebookInitFunc, data *mat64.Dense) (*Map, error) {
	// validate the map configuration
	if err := validateConfig(c); err != nil {
		return nil, err
	}
	// if nil codebook init function is passed in, use random init
	if initFunc == nil {
		return nil, fmt.Errorf("Invalid codebook init function: %v\n", initFunc)
	}
	// if input data is empty throw error
	if data == nil {
		return nil, fmt.Errorf("Invalid input data: %v\n", data)
	}
	// count the number of map units
	// TODO: figure out dims if Dims is 0,0
	mapUnits := 1
	for _, dim := range c.Dims {
		mapUnits = mapUnits * dim
	}
	// initialize codebook
	codebook, err := initFunc(data, mapUnits)
	if err != nil {
		return nil, fmt.Errorf("Failed to initialize codebook: %s\n", err)
	}
	// return pointer to new map
	return &Map{
		codebook: codebook,
	}, nil
}

// Codebook returns a matrix which contains SOM codebook vectors
func (m Map) Codebook() *mat64.Dense {
	return m.codebook
}

// UDist returns a matrix which contains Euclidean distances between SOM units
func (m Map) UDist() *mat64.Dense {
	return m.uDist
}

// UCoords returns a matrix which contains SOM unit coordinates on the map
func (m Map) UCoords() *mat64.Dense {
	return m.uCoords
}

// Bmus returns a matrix which contains SOM indices of Best Match Units (BMUs)
func (m Map) Bmus() []int {
	return m.bmus
}
