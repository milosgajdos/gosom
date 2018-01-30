package som

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// uShapes maps supported SOM unit shapes
var uShapes = map[string]bool{
	"hexagon":   true,
	"rectangle": true,
}

// gridTypes maps supported grid types
var coordsInitFns = map[string]coordsInitFunc{
	"planar": GridCoords,
}

// decays maps supported decay strategies
var decays = map[string]bool{
	"lin": true,
	"exp": true,
	"inv": true,
}

// trainings maps supported training algorithms
var trainingAlgs = map[string]bool{
	"seq":   true,
	"batch": true,
}

// coordsInitFunc defines SOM grid coordinates initialization function
type coordsInitFunc func(string, []int) (*mat.Dense, error)

// NeighbFunc defines SOM neighbourhood function
type NeighbFunc func(float64, float64) float64

// CbInitFunc defines SOM codebook initialization function
type CbInitFunc func(*mat.Dense, []int) (*mat.Dense, error)

// GridConfig holds SOM grid configuration
type GridConfig struct {
	// Size specifies SOM grid dimensions
	Size []int
	// Type specifies the type of SOM grid: planar
	Type string
	// UShape specifies SOM unit shape: hexagon, rectangle
	UShape string
}

// CbConfig holds SOM codebook configuration
type CbConfig struct {
	// Dim defines number of codebook vector dimension
	Dim int
	// InitFunc specifies codebook initialization function
	InitFunc CbInitFunc
}

// MapConfig holds SOM configuration
type MapConfig struct {
	// Grid is SOM grid config configuration
	Grid *GridConfig
	// Codebook holds SOM codebook configuration
	Cb *CbConfig
}

// TrainConfig holds SOM training configuration
type TrainConfig struct {
	// Algorithm specifies training method: seq or batch
	Algorithm string
	// Radius specifies initial SOM units radius
	Radius float64
	// RDecay specifies radius decay strategy: lin, exp
	RDecay string
	// NeighbFn specifies SOM neighbourhood function: gaussian, bubble, mexican
	NeighbFn NeighbFunc
	// LRate specifies initial SOM learning rate
	LRate float64
	// LDecay specifies learning rate decay strategy: lin, exp
	LDecay string
}

// validateGridConfig validates SOM grid configuration
// It returns error if any of the config parameters are invalid
func validateGridConfig(c *GridConfig) error {
	// SOM must have 2 dimensions
	// TODO: figure out 3D maps
	if len(c.Size) != 2 {
		return fmt.Errorf("unsupported number of SOM grid dimensions supplied: %d", len(c.Size))
	}
	// check if the supplied dimensions are negative integers or if they are single node
	product := 1
	for _, dim := range c.Size {
		if dim <= 0 {
			return fmt.Errorf("incorrect SOM grid dimensions supplied: %v", c.Size)
		}
		product *= dim
	}
	// 1D dimensions supplied: [1,1,1...]
	if product == 1 {
		return fmt.Errorf("incorrect SOM grid dimensions supplied: %v", c.Size)
	}
	// check if the supplied grid type is supported
	if _, ok := coordsInitFns[c.Type]; !ok {
		return fmt.Errorf("unsupported SOM grid type: %s", c.Type)
	}
	// check if the supplied unit shape type is supported
	if _, ok := uShapes[c.UShape]; !ok {
		return fmt.Errorf("unsupported SOM unit shape: %s", c.UShape)
	}

	return nil
}

// validateCbConfig validates SOM configuration.
// It returns error if any of the config parameters are invalid
func validateCbConfig(c *CbConfig) error {
	// codebook vectors must have non-zero dimensions
	if c.Dim <= 0 {
		return fmt.Errorf("incorrect SOM codebook dimension supplied: %v", c.Dim)
	}
	// check if the codebook init func is not nil
	if c.InitFunc == nil {
		return fmt.Errorf("invalid InitFunc: %v", c.InitFunc)
	}
	return nil
}

// validateTrainConfig validtes SOM training configuration
// It returns error if any of the training config parameters are invalid
func validateTrainConfig(c *TrainConfig) error {
	// training method must be supported
	if _, ok := trainingAlgs[c.Algorithm]; !ok {
		return fmt.Errorf("invalid SOM training algorithm: %s", c.Algorithm)
	}
	// initial SOM unit radius must be greater than zero
	if c.Radius < 0 {
		return fmt.Errorf("invalid SOM unit radius: %f", c.Radius)
	}
	// check Radius decay strategy
	if _, ok := decays[c.RDecay]; !ok {
		return fmt.Errorf("unsupported Radius decay strategy: %s", c.RDecay)
	}
	// check the supplied is not nil
	if c.NeighbFn == nil {
		return fmt.Errorf("invalid Neighbourhood function: %v", c.NeighbFn)
	}
	// initial SOM learning rate must be greater than zero
	if c.LRate < 0 {
		return fmt.Errorf("invalid SOM learning rate: %f", c.LRate)
	}
	// check Learning rate decay strategy
	if _, ok := decays[c.LDecay]; !ok {
		return fmt.Errorf("unsupported Learning rate decay strategy: %s", c.LDecay)
	}
	return nil
}
