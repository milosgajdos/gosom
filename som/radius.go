package som

import "math"

// Radius is a decay function for the radius parameter.  The supported strategies
// are "exp" and "lin".  "exp" is an exponential decay function, "lin" is linear.
// At iteration 0 the function returns the radius0, at totalIterations it returns 1.0
func Radius(iteration, totalIterations int, strategy string, radius0 float64) float64 {
	switch strategy {
	case "exp":
		return expRadius(iteration, totalIterations, radius0)
	case "lin":
		return linRadius(iteration, totalIterations, radius0)
	default:
		return expRadius(iteration, totalIterations, radius0)
	}
}

func expRadius(iteration, totalIterations int, radius0 float64) float64 {
	lambda := float64(totalIterations) / math.Log(radius0)
	return radius0 * math.Exp(-float64(iteration)/lambda)
}

func linRadius(iteration, totalIterations int, radius0 float64) float64 {
	return radius0 - float64(iteration)/float64(totalIterations)*(radius0-1)
}
