package som

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// UShape contains supported SOM unit shapes
var UShape = map[string]bool{
	"hexagon":   true,
	"rectangle": true,
}

// CoordsInit maps supported grid coordinates function types to their implementations
var CoordsInit = map[string]CoordsInitFunc{
	"planar": GridCoords,
}

// Neighb maps supported neighbourhood functions to their implementations
var Neighb = map[string]NeighbFunc{
	"gaussian": Gaussian,
	"bubble":   Bubble,
	"mexican":  Mexican,
}

// Decay maps supported decay strategies
var Decay = map[string]bool{
	"lin": true,
	"exp": true,
	"inv": true,
}

// Training maps supported training methods
var Training = map[string]bool{
	"seq":   true,
	"batch": true,
}

// CodebookInitFunc defines SOM codebook initialization function
type CodebookInitFunc func(*mat64.Dense, []int) (*mat64.Dense, error)

// CoordsInitFunc defines SOM grid coordinates initialization function
type CoordsInitFunc func(string, []int) (*mat64.Dense, error)

// NeighbFunc defines SOM neighbourhood function
type NeighbFunc func(float64, float64) float64

// GridConfig holds SOM grid configuration
type GridConfig struct {
	// Dims specifies SOM grid dimensions
	Dims []int
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
	InitFunc CodebookInitFunc
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
	// Method specifies training method: seq or batch
	Method string
	// Radius specifies initial SOM units radius
	Radius float64
	// RDecay specifies radius decay strategy: lin, exp
	RDecay string
	// NeighbFn specifies SOM neighbourhood function: gaussian, bubble, mexican
	NeighbFn string
	// LRate specifies initial SOM learning rate
	LRate float64
	// LDecay specifies learning rate decay strategy: lin, exp
	LDecay string
}

// validateMapConfig validates SOm configuration
// It returns error or if any of the configuration parameters are invalid
func validateMapConfig(c *MapConfig) error {
	// validate grid configuration
	if err := validateGridConfig(c.Grid); err != nil {
		return err
	}
	// validate codebook configuration
	if err := validateCbConfig(c.Cb); err != nil {
		return err
	}

	return nil
}

// validateGridConfig validates SOM grid configuration
// It returns error if any of the config parameters are invalid
func validateGridConfig(c *GridConfig) error {
	// SOM must have 2 dimensions
	// TODO: figure out 3D maps
	if len(c.Dims) != 2 {
		return fmt.Errorf("unsupported number of SOM grid dimensions supplied: %d", len(c.Dims))
	}
	// check if the supplied dimensions are negative integers or if they are single node
	prod := 1
	for _, dim := range c.Dims {
		if dim <= 0 {
			return fmt.Errorf("incorrect SOM grid dimensions supplied: %v", c.Dims)
		}
		prod *= dim
	}
	if prod == 1 {
		return fmt.Errorf("incorrect SOM grid dimensions supplied: %v", c.Dims)
	}
	// check if the supplied grid type is supported
	if _, ok := CoordsInit[c.Type]; !ok {
		return fmt.Errorf("unsupported SOM grid type: %s", c.Type)
	}
	// check if the supplied unit shape type is supported
	if _, ok := UShape[c.UShape]; !ok {
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
	if _, ok := Training[c.Method]; !ok {
		return fmt.Errorf("invalid SOM training method: %s", c.Method)
	}
	// initial SOM unit radius must be greater than zero
	if c.Radius < 0 {
		return fmt.Errorf("invalid SOM unit radius: %f", c.Radius)
	}
	// check Radius decay strategy
	if _, ok := Decay[c.RDecay]; !ok {
		return fmt.Errorf("unsupported Radius decay strategy: %s", c.RDecay)
	}
	// check the supplied neighbourhood function
	if _, ok := Neighb[c.NeighbFn]; !ok {
		return fmt.Errorf("unsupported Neighbourhood function: %s", c.NeighbFn)
	}
	// initial SOM learning rate must be greater than zero
	if c.LRate < 0 {
		return fmt.Errorf("invalid SOM learning rate: %f", c.LRate)
	}
	// check Learning rate decay strategy
	if _, ok := Decay[c.LDecay]; !ok {
		return fmt.Errorf("unsupported Learning rate decay strategy: %s", c.LDecay)
	}
	return nil
}
