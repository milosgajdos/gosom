package som

import (
	"fmt"
	"math"
)

const SmallestRate = 0.01

// LearningRate is a decay function with two available strategies: exponential and linear.
// Both strategies return the learningRate0 for iteration 0 and SmallestRate
// for iteration == totalIterations-1
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
	lambda := float64(totalIterations-1) / math.Log(learningRate0/SmallestRate)
	return learningRate0 * math.Exp(-float64(iteration)/lambda)
}

func linLR(iteration, totalIterations int, learningRate0 float64) float64 {
	return learningRate0 - float64(iteration)/float64(totalIterations-1)*(learningRate0-SmallestRate)
}
