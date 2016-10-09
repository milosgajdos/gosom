package som

import "math"

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
	return radius0 - float64(iteration)/float64(totalIteration)*(radius0-1)
}
