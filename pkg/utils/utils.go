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
	// if empty string, return empty slice
	if dimString == "" {
		return make([]int, 2), nil
	}

	// nolint:prealloc
	var intDims []int
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

// IntProduct calculates a product of all items in slice passed in as a parameter
// It returns 1.0 if the slice is nil or empty.
func IntProduct(vector []int) int {
	if len(vector) == 0 || vector == nil {
		return 1
	}
	product := 1
	for _, v := range vector {
		product = product * v
	}
	return product
}

// IntCumProduct calculates a cumulative product of all items in slice as a parameter
// and returns it in a new slice. The origin slice remains unmodified
// It returns empty slice if either the supplied slice is empty or nil
func IntCumProduct(vector []int) []int {
	cumProd := []int{}
	if len(vector) == 0 || vector == nil {
		return cumProd
	}
	partProd := 1
	for _, val := range vector {
		partProd = partProd * val
		cumProd = append(cumProd, partProd)
	}
	return cumProd
}
