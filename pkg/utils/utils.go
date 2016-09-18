package utils

import (
	"strconv"
	"strings"
)

// ParseDims expects a string that contains comma-separated integer values.
// It tries to convert the parameters to their numerical values and returns them.
// If empty string is supplied, it allocates 2-dims slice and returns it.
// It returns error if a non-empty string is passed in as a parameter and
// the values it contains can not be converted to integers.
func ParseDims(dimString string) ([]int, error) {
	var intDims []int
	// if empty string, return empty slice
	if dimString == "" {
		intDims = make([]int, 2)
		return intDims, nil
	}
	// parse non-empty string into int slice
	strDims := strings.Split(dimString, ",")
	for _, strDim := range strDims {
		intDim, err := strconv.Atoi(strDim)
		if err != nil {
			return nil, err
		}
		intDims = append(intDims, intDim)
	}
	return intDims, nil
}
