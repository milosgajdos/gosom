package som

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestDistance(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		a        []float64
		b        []float64
		expected float64
	}{
		{[]float64{0.0, 0.0}, []float64{0.0, 1.0}, 1.0},
		{[]float64{0.0, 0.0}, []float64{0.0, 0.0}, 0.0},
		{[]float64{3.0, 1.0}, []float64{1.0, 3.0}, 2.828},
	}

	for _, tc := range testCases {
		a := mat64.NewVector(len(tc.a), tc.a)
		b := mat64.NewVector(len(tc.b), tc.b)
		dist, err := Distance("euclidean", a, b)
		assert.NoError(err)
		assert.InDelta(tc.expected, dist, 0.01)
	}

	// nil vectors
	d, err := Distance("euclidean", nil, nil)
	assert.Error(err)
	assert.Equal(0.0, d)
	// different vector dimensions
	a := mat64.NewVector(2, []float64{0.0, 0.0})
	b := mat64.NewVector(1, []float64{1.0})
	d, err = Distance("euclidean", a, b)
	assert.Error(err)
	assert.Equal(0.0, d)
}

func TestDistanceMx(t *testing.T) {
	assert := assert.New(t)

	one := mat64.NewDense(2, 2, []float64{
		0.0, 0.0,
		1.0, 0.0,
	})

	oneR, _ := one.Dims()
	oneOutExpected := mat64.NewDense(oneR, oneR, []float64{
		0.0, 1.0,
		1.0, 0.0,
	})

	oneOut, err := DistanceMx("euclidean", one)

	assert.NoError(err)
	assert.True(mat64.EqualApprox(oneOutExpected, oneOut, 0.01))

	zero := mat64.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		33.0, 33.0, 33.0,
	})

	zeroR, _ := zero.Dims()
	zeroOutExpected := mat64.NewDense(zeroR, zeroR, []float64{
		0.0, 0.0,
		0.0, 0.0,
	})

	zeroOut, err := DistanceMx("euclidean", zero)

	assert.NoError(err)
	assert.True(mat64.EqualApprox(zeroOutExpected, zeroOut, 0.01))

	negative := mat64.NewDense(2, 3, []float64{
		33.0, 33.0, 33.0,
		133.0, 33.0, 33.0,
	})

	negativeR, _ := negative.Dims()
	negativeOutExpected := mat64.NewDense(negativeR, negativeR, []float64{
		0.0, 100.0,
		100.0, 0.0,
	})

	negativeOut, err := DistanceMx("euclidean", negative)

	assert.NoError(err)
	assert.True(mat64.EqualApprox(negativeOutExpected, negativeOut, 0.01))

	nilMatrix, err := DistanceMx("euclidean", nil)

	assert.Error(err)
	assert.Nil(nilMatrix)
}

func TestClosestVec(t *testing.T) {
	assert := assert.New(t)

	metric := "euclidean"
	testCases := []struct {
		v        []float64
		m        []float64
		metric   string
		expected int
	}{
		{[]float64{0.0, 0.0}, []float64{0.0, 1.0, 0.0, 0.1}, metric, 1},
		{[]float64{0.0, 0.0}, []float64{0.0, 0.0, 0.0, 0.1}, metric, 0},
		{[]float64{3.0, 1.0}, []float64{1.0, 3.0, 1.0, 0.0}, metric, 1},
	}

	for _, tc := range testCases {
		v := mat64.NewVector(len(tc.v), tc.v)
		m := mat64.NewDense(2, len(tc.v), tc.m)
		closest, err := ClosestVec(tc.metric, v, m)
		assert.NoError(err)
		assert.Equal(tc.expected, closest)
	}

	// nil matrix returns error
	v := new(mat64.Vector)
	m := new(mat64.Dense)
	closest, err := ClosestVec(metric, v, nil)
	assert.Error(err)
	assert.Equal(-1, closest)
	// nil vector returns error
	closest, err = ClosestVec(metric, nil, m)
	assert.Error(err)
	assert.Equal(-1, closest)
}
