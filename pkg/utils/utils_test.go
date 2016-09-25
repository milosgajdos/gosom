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

func TestProduct(t *testing.T) {
	assert := assert.New(t)

	testFloatCases := []struct {
		vector   []int
		expected int
	}{
		{nil, 1},
		{[]int{}, 1},
		{[]int{1, 2, 3}, 6},
	}

	for _, tc := range testFloatCases {
		p := IntProduct(tc.vector)
		assert.EqualValues(p, tc.expected)
	}
}

func TestCumProduct(t *testing.T) {
	assert := assert.New(t)

	testFloatCases := []struct {
		vector   []int
		expected []int
	}{
		{nil, []int{}},
		{[]int{}, []int{}},
		{[]int{1, 2, 3}, []int{1, 2, 6}},
	}

	for _, tc := range testFloatCases {
		p := IntCumProduct(tc.vector)
		assert.EqualValues(p, tc.expected)
	}
}
