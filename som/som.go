package som

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/pkg/utils"
)

// CodebookInitFunc defines SOM codebook initialization function
type CodebookInitFunc func(*mat64.Dense, int, []int) (*mat64.Dense, error)

// CoordsInitFunc defines SOM grid coordinates initialization function
type CoordsInitFunc func(string, []int) (*mat64.Dense, error)

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

// NewMap creates new SOM based on the provided configuration and input data
// NewMap allows you to pass in SOM codebook init function that is used to initialize
// SOM codebook vectors to initial values. If initFunc is nil, random initializatio is used.
// It returns error if the provided configuration is not valid or if the data matrix
// passed in as a parameter is empty or if the codebook could not be initialized.
func NewMap(c *Config, initFunc CodebookInitFunc, data *mat64.Dense) (*Map, error) {
	// validate the map configuration
	if err := validateConfig(c); err != nil {
		return nil, err
	}
	// if nil codebook init function is passed in, use random init
	if initFunc == nil {
		initFunc = RandInit
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
	codebook, err := initFunc(data, mUnits, c.Dims)
	if err != nil {
		return nil, fmt.Errorf("Failed to initialize codebook: %s\n", err)
	}
	// codebook vectors must have same number of dimensions as data
	_, dataDim := data.Dims()
	_, cbCols := codebook.Dims()
	if cbCols != dataDim {
		return nil, fmt.Errorf("")
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
