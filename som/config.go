package som

import "fmt"

// CodebookInit maps SOM init function types to itheir actual implementations
var CodebookInit = map[string]CodebookInitFunc{
	"linear": LinInit,
	"random": RandInit,
}

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

// Cool maps supported cooling strategies
var Cool = map[string]bool{
	"lin": true,
	"exp": true,
	"inv": true,
}

// Config holds SOM configuration
type Config struct {
	// Dims specifies SOM dimensions
	Dims []int
	// Grid specifies the type of SOM grid: planar
	Grid string
	// UShape specifies SOM unit shape: hexagon, rectangle
	UShape string
	// Radius specifies initial SOM units radius
	Radius int
	// RCool specifies radius cooling strategy: lin, exp
	RCool string
	// NeighbFn specifies SOM neighbourhood function: gaussian, bubble, mexican
	NeighbFn string
	// LRate specifies initial SOM learning rate
	LRate int
	// LCool specifies learning rate cooling strategy: lin, exp
	LCool string
}

// validateConfig validates SOM configuration.
// It returns error if any of the config parameters are invalid
func validateConfig(c *Config) error {
	// SOM must have 2 dimensions
	// TODO: figure out 3D maps
	if dimLen := len(c.Dims); dimLen != 2 {
		return fmt.Errorf("Incorrect number of dimensions supplied: %d\n", dimLen)
	}
	// check if the supplied dimensions are negative integers
	for _, dim := range c.Dims {
		if dim < 0 {
			return fmt.Errorf("Incorrect SOM dimensions supplied: %v\n", c.Dims)
		}
	}
	// check if the supplied grid type is supported
	if _, ok := CoordsInit[c.Grid]; !ok {
		return fmt.Errorf("Unsupported SOM grid type: %s\n", c.Grid)
	}
	// check if the supplied unit shape type is supported
	if _, ok := UShape[c.UShape]; !ok {
		return fmt.Errorf("Unsupported SOM unit shape: %s\n", c.UShape)
	}
	// initial SOM unit radius must be greater than zero
	if c.Radius < 0 {
		return fmt.Errorf("Invalid SOM unit radius: %d\n", c.Radius)
	}
	// check Radius cooling strategy
	if _, ok := Cool[c.RCool]; !ok {
		return fmt.Errorf("Unsupported Radius cooling strategy: %s\n", c.RCool)
	}
	// hcheck the supplied neighbourhood function
	if _, ok := Neighb[c.NeighbFn]; !ok {
		return fmt.Errorf("Unsupported Neighbourhood function: %s\n", c.NeighbFn)
	}
	// initial SOM learning rate must be greater than zero
	if c.LRate < 0 {
		return fmt.Errorf("Invalid SOM learning rate: %d\n", c.LRate)
	}
	// check Learning rate cooling strategy
	if _, ok := Cool[c.LCool]; !ok {
		return fmt.Errorf("Unsupported Learning rate cooling strategy: %s\n", c.LCool)
	}
	return nil
}
