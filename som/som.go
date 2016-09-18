package som

import "github.com/gonum/matrix/mat64"

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

// NewMap creates new SOM based on the provided configuration and data stored in a matrix.
// It fails with error if the provided configuration is not valid or data matrix passed in is empty.
func NewMap(c *Config, d mat64.Matrix) (*Map, error) {
	// validate the map configuration
	if err := validateConfig(c); err != nil {
		return nil, err
	}
	return &Map{}, nil
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
