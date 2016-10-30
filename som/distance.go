package som

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// Distance calculates a distance metric between vectors a and b.
// If unsupported metric is requested DistVector returns euclidean distance.
// It returns error if the supplied vectors are either nil or are different dimensions
func Distance(metric string, a, b *mat64.Vector) (float64, error) {
	if a == nil || b == nil {
		return 0.0, fmt.Errorf("Invalid vectors supplied. a: %v, b: %v\n", a, b)
	}
	if a.Len() != b.Len() {
		return 0.0, fmt.Errorf("Incorrect vector dims. a: %d, b: %d\n", a.Len(), b.Len())
	}

	switch metric {
	case "euclidean":
		return euclideanVec(a, b), nil
	default:
		return euclideanVec(a, b), nil
	}
}

// DistanceMx calculates a distance matrix for the given matrix using the given metric.
// Distance matrix is also known in literature as dissimilarity matrix.
// It returns a hollow symmetric matrix where an item x_ij contains the distance between
// vectors stored in rows i and j. You can request different distance metrics via metric
// parameter. If an unknown metric is supplied Euclidean distance is computed.
// It returns error if the supplied matrix is nil.
func DistanceMx(metric string, m *mat64.Dense) (*mat64.Dense, error) {
	if m == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", m)
	}

	switch metric {
	case "euclidean":
		return euclideanMx(m), nil
	default:
		return euclideanMx(m), nil
	}
}

// ClosestVec finds the closest vector to v from the vectors stored in m rows
// using the requested distance metric. It returns an index to the m rows.
// If several vectors of the same distance are found, it returns the index of the first one.
// ClosestVec returns error if either v or m are nil or if the distance could not be calculated.
// When the ClosestVec fails with error, returned index is set to -1.
func ClosestVec(metric string, v *mat64.Vector, m *mat64.Dense) (int, error) {
	// vector can't be nil
	if v == nil {
		return -1, fmt.Errorf("Invalid vector: %v\n", v)
	}
	// matrix cant be nil
	if m == nil {
		return -1, fmt.Errorf("Invalid matrix: %v\n", m)
	}
	// check if the dimensions are ok
	rows, _ := m.Dims()
	closest := 0
	dist := math.MaxFloat64
	for i := 0; i < rows; i++ {
		d, err := Distance(metric, v, m.RowView(i))
		if err != nil {
			return -1, err
		}
		if d < dist {
			dist = d
			closest = i
		}
	}

	return closest, nil
}

// euclideanVec computes euclidean distance between vectors a and b.
func euclideanVec(a, b *mat64.Vector) float64 {
	// to optimize distance calculations we use raw data
	aData := a.RawVector().Data
	bData := b.RawVector().Data
	d := 0.0
	for i := 0; i < a.Len(); i++ {
		d += (aData[i] - bData[i]) * (aData[i] - bData[i])
	}
	return math.Sqrt(d)
}

// euclideanMx computes a matrix of euclidean distances between each row in m
func euclideanMx(m *mat64.Dense) *mat64.Dense {
	rows, _ := m.Dims()
	out := mat64.NewDense(rows, rows, nil)

	for row := 0; row < rows-1; row++ {
		dist := 0.0
		a := m.RowView(row)
		for i := row + 1; i < rows; i++ {
			if i != row {
				b := m.RowView(i)
				dist = euclideanVec(a, b)
				out.Set(row, i, dist)
				out.Set(i, row, dist)
			}
		}
	}

	return out
}
