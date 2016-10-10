package som

import (
	"fmt"
	"math"
)

const SmallestRadius = 1.0

// Radius is a decay function for the radius parameter.  The supported strategies
// are "exp" and "lin".  "exp" is an exponential decay function, "lin" is linear.
// At iteration 0 the function returns the radius0, at totalIterations-1 it returns SmallestRadius
// radius0 has to be a positive number
func Radius(iteration, totalIterations int, strategy string, radius0 float64) (float64, error) {
	if radius0 <= 0.0 {
		return math.NaN(), fmt.Errorf("radius0 must be a positive number")
	}
	switch strategy {
	case "exp":
		return expRadius(iteration, totalIterations, radius0), nil
	case "lin":
		return linRadius(iteration, totalIterations, radius0), nil
	default:
		return expRadius(iteration, totalIterations, radius0), nil
	}
}

func expRadius(iteration, totalIterations int, radius0 float64) float64 {
	lambda := float64(totalIterations-1) / math.Log(radius0/SmallestRadius)
	return radius0 * math.Exp(-float64(iteration)/lambda)
}

func linRadius(iteration, totalIterations int, radius0 float64) float64 {
	return radius0 - float64(iteration)/float64(totalIterations-1)*(radius0-SmallestRadius)
}
