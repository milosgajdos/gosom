package som

import "math"

// Gaussian calculates gaussian neghbourhood
func Gaussian(distance float64, radius float64) float64 {
	return math.Exp(-(distance * distance) / (2 * radius * radius))
}

// Bubble calculates bubble neghbourhood
func Bubble(distance float64, radius float64) float64 {
	if distance <= radius {
		return 1.0
	}
	return 0.0
}

// MexicanHat calculates mexican hat neghbourhood
func MexicanHat(distance float64, radius float64) float64 {
	return 2 / (math.Sqrt(3*radius) * math.Pow(math.Pi, 0.25)) *
		(1 - (distance*distance)/(radius*radius)) *
		math.Exp(-(distance*distance)/(2*radius*radius))
}
