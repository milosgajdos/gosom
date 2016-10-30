package som

import (
	"fmt"
	"math"
)

// MinRadius is the smallest allowed SOM unit Radius
const MinRadius = 1.0

// Radius is a decay function for the SOM neighbourhood radius parameter.
// It supports exponential and linear decay strategies denoted as "exp" and "lin".
// Any other strategy defaults to "exp". At the first iteration the function returns
// the initRadius, at totalIterations-1 it returns MinRadius.
// It returns error if initRadius is not a positive integer
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
