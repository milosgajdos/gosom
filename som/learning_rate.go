package som

import (
	"fmt"
	"math"
)

// MinLRate smallest possible learning rate
const MinLRate = 0.01

// LRate is a decay function for the LRate parameter. The supported strategies are "exp" and "lin"
// Any other strategy defaults to "exp". "exp" is an exponential decay function, "lin" is linear.
// At iteration 0 the function returns the initLRate, at totalIterations-1 it returns MinLRate
// initLRate has to be a positive number
func LRate(iteration, totalIterations int, strategy string, initLRate float64) (float64, error) {
	if initLRate <= 0.0 {
		return math.NaN(), fmt.Errorf("initLRate must be a positive number")
	}

	switch strategy {
	case "exp":
		return expLRate(iteration, totalIterations, initLRate), nil
	case "lin":
		return linLRate(iteration, totalIterations, initLRate), nil
	default:
		return expLRate(iteration, totalIterations, initLRate), nil
	}
}

func expLRate(iteration, totalIterations int, initLRate float64) float64 {
	lambda := float64(totalIterations-1) / math.Log(initLRate/MinLRate)
	return initLRate * math.Exp(-float64(iteration)/lambda)
}

func linLRate(iteration, totalIterations int, initLRate float64) float64 {
	return initLRate - float64(iteration)/float64(totalIterations-1)*(initLRate-MinLRate)
}
