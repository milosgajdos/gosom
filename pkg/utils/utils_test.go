package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseDims(t *testing.T) {
	assert := assert.New(t)
	// test cases
	testCases := []struct {
		dimString string
		expected  []int
		err       bool
	}{
		{"", []int{0, 0}, false},
		{"one,two", nil, true},
		{"1,2,", nil, true},
		{"1,2", []int{1, 2}, false},
		{"1,2,3", []int{1, 2, 3}, false},
	}

	for _, tc := range testCases {
		intDims, err := ParseDims(tc.dimString)
		if tc.err {
			assert.Error(err)
		} else {
			assert.NoError(err)
			assert.Exactly(tc.expected, intDims)
		}
	}
}
