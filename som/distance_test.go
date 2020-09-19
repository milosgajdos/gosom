package som

import (
	"fmt"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestDistance(t *testing.T) {
	assert := assert.New(t)

	metric := Euclidean
	testCases := []struct {
		metric   Metric
		a        []float64
		b        []float64
		expected float64
	}{
		{metric, []float64{0.0, 0.0}, []float64{0.0, 1.0}, 1.0},
		{metric, []float64{0.0, 0.0}, []float64{0.0, 0.0}, 0.0},
		{metric, []float64{3.0, 1.0}, []float64{1.0, 3.0}, 2.828},
	}

	for _, tc := range testCases {
		dist, err := Distance(tc.metric, tc.a, tc.b)
		assert.NoError(err)
		assert.InDelta(tc.expected, dist, 0.01)
	}

	// unknown metric returns euclidean distance
	a := []float64{0.0, 0.0}
	b := []float64{0.0, 1.0}
	d, err := Distance(Metric(1000), a, b)
	assert.NoError(err)
	assert.InDelta(1.0, d, 0.01)
	// nil vectors
	d, err = Distance(metric, nil, nil)
	assert.Error(err)
	assert.Equal(0.0, d)
	// different vector dimensions
	a = []float64{0.0, 0.0}
	b = []float64{1.0}
	d, err = Distance(metric, a, b)
	assert.Error(err)
	assert.Equal(0.0, d)
}

func TestDistanceMx(t *testing.T) {
	assert := assert.New(t)

	metric := Euclidean
	one := mat.NewDense(2, 2, []float64{
		0.0, 0.0,
		1.0, 0.0,
	})

	oneR, _ := one.Dims()
	oneOutExpected := mat.NewDense(oneR, oneR, []float64{
		0.0, 1.0,
		1.0, 0.0,
	})

	oneOut, err := DistanceMx(metric, one)

	assert.NoError(err)
	assert.True(mat.EqualApprox(oneOutExpected, oneOut, 0.01))

	// test if default distance is computed for unknown matrix
	oneOut, err = DistanceMx(Metric(1000), one)

	assert.NoError(err)
	assert.True(mat.EqualApprox(oneOutExpected, oneOut, 0.01))

	zero := mat.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		33.0, 33.0, 33.0,
	})

	zeroR, _ := zero.Dims()
	zeroOutExpected := mat.NewDense(zeroR, zeroR, []float64{
		0.0, 0.0,
		0.0, 0.0,
	})

	zeroOut, err := DistanceMx(metric, zero)

	assert.NoError(err)
	assert.True(mat.EqualApprox(zeroOutExpected, zeroOut, 0.01))

	negative := mat.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		133.0, 33.0, 33.0,
	})

	negativeR, _ := negative.Dims()
	negativeOutExpected := mat.NewDense(negativeR, negativeR, []float64{
		0.0, 100.0,
		100.0, 0.0,
	})

	negativeOut, err := DistanceMx(metric, negative)

	assert.NoError(err)
	assert.True(mat.EqualApprox(negativeOutExpected, negativeOut, 0.01))

	nilMatrix, err := DistanceMx(metric, nil)

	assert.Error(err)
	assert.Nil(nilMatrix)
}

func TestClosestVec(t *testing.T) {
	assert := assert.New(t)

	metric := Euclidean
	testCases := []struct {
		metric   Metric
		v        []float64
		m        []float64
		expected int
	}{
		{metric, []float64{0.0, 0.0}, []float64{0.0, 1.0, 0.0, 0.1}, 1},
		{metric, []float64{0.0, 0.0}, []float64{0.0, 0.0, 0.0, 0.1}, 0},
		{metric, []float64{3.0, 1.0}, []float64{1.0, 3.0, 1.0, 0.0}, 1},
	}

	for _, tc := range testCases {
		m := mat.NewDense(2, len(tc.v), tc.m)
		closest, err := ClosestVec(tc.metric, tc.v, m)
		assert.NoError(err)
		assert.Equal(tc.expected, closest)
	}

	// nil vector returns error
	v := []float64{}
	m := new(mat.Dense)
	errString := "invalid vector: %v"
	closest, err := ClosestVec(metric, v, m)
	assert.Error(err)
	assert.EqualError(err, fmt.Sprintf(errString, v))
	assert.Equal(-1, closest)
	// nil matrix returns error
	v = []float64{1.0}
	m = nil
	errString = "invalid matrix: %v"
	closest, err = ClosestVec(metric, v, m)
	assert.Error(err)
	assert.EqualError(err, fmt.Sprintf(errString, m))
	assert.Equal(-1, closest)
	// mismatched dimensions return error
	v = make([]float64, 3)
	m = mat.NewDense(2, 2, nil)
	closest, err = ClosestVec(metric, v, m)
	assert.Error(err)
	assert.Equal(-1, closest)
}

func TestClosestNVec(t *testing.T) {
	assert := assert.New(t)

	metric := Euclidean
	// test failure cases
	v := []float64{}
	m := new(mat.Dense)
	n := 2
	// nil vector returns error
	errString := "invalid vector: %v"
	closest, err := ClosestNVec(metric, n, v, m)
	assert.EqualError(err, fmt.Sprintf(errString, v))
	assert.Nil(closest)
	// nil matrix returns error
	v = []float64{1.0}
	m = nil
	errString = "invalid matrix: %v"
	closest, err = ClosestNVec(metric, n, v, m)
	assert.EqualError(err, fmt.Sprintf(errString, m))
	// incorrect number of n closest vectors
	m = new(mat.Dense)
	n = -5
	errString = "invalid number of closest vectors requested: %d"
	closest, err = ClosestNVec(metric, n, v, m)
	assert.EqualError(err, fmt.Sprintf(errString, n))
	assert.Nil(closest)
	// when n==1, return BMU
	n = 1
	v, mData := []float64{0.0, 0.0}, []float64{0.0, 1.0, 0.0, 0.1}
	m = mat.NewDense(2, len(v), mData)
	closest, err = ClosestNVec(metric, n, v, m)
	assert.NotNil(closest)
	assert.Equal(1, closest[0])
	// find 2 closest vectors
	n = 2
	mData = []float64{
		0.0, 1.0,
		0.0, 0.1,
		0.0, 0.2,
		0.1, 0.0,
		0.0, 0.5}
	m = mat.NewDense(5, len(v), mData)
	closest, err = ClosestNVec(metric, n, v, m)
	assert.NoError(err)
	sort.Ints(closest)
	assert.EqualValues([]int{1, 3}, closest)
}

func TestBMUs(t *testing.T) {
	assert := assert.New(t)

	// test data and codebook
	rows := 3
	data := mat.NewDense(rows, 4,
		[]float64{5.1, 3.5, 1.4, 0.1,
			4.6, 3.1, 1.5, 0.4,
			5.0, 3.6, 1.4, 0.5})
	cbook := mat.NewDense(2, 4,
		[]float64{5.1, 3.5, 1.4, 0.1,
			5.0, 3.6, 1.4, 0.5})
	// nil data returns error
	errString := "invalid data supplied: %v"
	bmus, err := BMUs(nil, cbook)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Nil(bmus)
	// nil codebook returns error
	errString = "invalid codebook supplied: %v"
	bmus, err = BMUs(data, nil)
	assert.EqualError(err, fmt.Sprintf(errString, nil))
	assert.Nil(bmus)
	// this should pass through without errors
	bmus, err = BMUs(data, cbook)
	assert.NoError(err)
	assert.NotNil(bmus)
	assert.Equal(rows, len(bmus))
}
