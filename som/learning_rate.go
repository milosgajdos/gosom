package som

import (
	"fmt"
	"math"
)

const SmallestLearningRate = 0.01

// LearningRate is a decay function for the learningRate parameter.  The supported strategies
// are "exp" and "lin" (any other defaults to "exp").  "exp" is an exponential decay function, "lin" is linear.
// At iteration 0 the function returns the learningRate0, at totalIterations-1 it returns SmallestLearningRate
// learningRate0 has to be a positive number
func LearningRate(iteration, totalIterations int, strategy string, learningRate0 float64) (float64, error) {
	if learningRate0 <= 0.0 {
		return math.NaN(), fmt.Errorf("learningRate0 must be a positive number")
	}

	switch strategy {
	case "exp":
		return expLR(iteration, totalIterations, learningRate0), nil
	case "lin":
		return linLR(iteration, totalIterations, learningRate0), nil
	default:
		return expLR(iteration, totalIterations, learningRate0), nil
	}
}

func expLR(iteration, totalIterations int, learningRate0 float64) float64 {
	lambda := float64(totalIterations-1) / math.Log(learningRate0/SmallestLearningRate)
	return learningRate0 * math.Exp(-float64(iteration)/lambda)
}

func linLR(iteration, totalIterations int, learningRate0 float64) float64 {
	return learningRate0 - float64(iteration)/float64(totalIterations-1)*(learningRate0-SmallestLearningRate)
}
