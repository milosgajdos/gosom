package som

import (
	"fmt"
	"math"
)

// MinRadius is the smallest allowed SOM unit Radius
const MinRadius = 1.0

// Radius is a decay function for the radius parameter. The supported strategies are "exp" and "lin".
// Any other strategy defaults to "exp". "exp" is an exponential decay function, "lin" is linear.
// At iteration 0 the function returns the initRadius, at totalIterations-1 it returns MinRadius
// initRadius has to be a positive number
func Radius(iteration, totalIterations int, strategy string, initRadius float64) (float64, error) {
	if initRadius <= 0.0 {
		return math.NaN(), fmt.Errorf("initRadius must be a positive number")
	}
	switch strategy {
	case "exp":
		return expRadius(iteration, totalIterations, initRadius), nil
	case "lin":
		return linRadius(iteration, totalIterations, initRadius), nil
	default:
		return expRadius(iteration, totalIterations, initRadius), nil
	}
}

func expRadius(iteration, totalIterations int, initRadius float64) float64 {
	lambda := float64(totalIterations-1) / math.Log(initRadius/MinRadius)
	return initRadius * math.Exp(-float64(iteration)/lambda)
}

func linRadius(iteration, totalIterations int, initRadius float64) float64 {
	return initRadius - float64(iteration)/float64(totalIterations-1)*(initRadius-MinRadius)
}
