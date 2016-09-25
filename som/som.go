package som

import (
	"fmt"
	"math"
	"strings"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/milosgajdos83/gosom/pkg/utils"
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
	mapUnits := utils.IntProduct(c.Dims)
	if mapUnits <= 1 {
		dims, err := gridDims(data, c.UShape)
		if err != nil {
			return nil, err
		}
		mapUnits = utils.IntProduct(dims)
		// set map dims to newl calculated dims
		c.Dims = dims
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

// gridDims tries to estimate the best dimensions of map from data matrix and given unit shape.
// It determines the grid size from eigenvectors of input data: the grid dimensions are
// calculated from the ratio of two highest input eigenvalues.
// It returns error if the map dimensions could not be calculated.
func gridDims(data *mat64.Dense, uShape string) ([]int, error) {
	dataLen, dataDim := data.Dims()
	// this is a simple heuristic - you can pick the scale > 5
	mapUnits := math.Ceil(5 * math.Sqrt(float64(dataLen)))
	// if the data is 1D - we return [1 x mapUnits] map dimensions
	if dataDim == 1 && dataLen > 1 {
		return []int{1, int(mapUnits)}, nil
	}
	// Not enough data to calculate eigenvectors
	// We will use heuristic: number of mapUnits = square area of SOM
	if dataLen < 2 {
		gDim := math.Sqrt(mapUnits)
		return []int{int(gDim), int(gDim)}, nil
	}
	// We have more than 2 samples and more than 1D data
	// Calculate eigenvalue ie. SVD singular values
	_, eigVals, ok := stat.PrincipalComponents(data, nil)
	if !ok {
		return nil, fmt.Errorf("Could not determine Principal Components")
	}
	// by default we use 1:1 ratio of the map
	ratio := 1.0
	// pick first two components: we only support 2D data maps
	// length check here is redundant, but let's make sure just in case
	if len(eigVals) >= 2 {
		if eigVals[0] != 0 && eigVals[1]*mapUnits >= eigVals[0] {
			ratio = math.Sqrt(eigVals[0] / eigVals[1])
		}
	}
	// If the unit shape is hexagon, the ratio is modified a bit to take it into account
	// Remember when using hexagon we don't get rectangle so the area != dimA * dimB
	tmpDim := math.Sqrt(mapUnits / ratio)
	if strings.EqualFold(uShape, "hexagon") {
		tmpDim = math.Sqrt(mapUnits / ratio * math.Sqrt(0.75))
	}
	yDim := int(floats.Min([]float64{mapUnits, tmpDim}))
	xDim := int(mapUnits / float64(yDim))
	// Return map dimensions
	return []int{xDim, yDim}, nil
}
