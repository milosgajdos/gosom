package som

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// Distance calculates a distance metric between vectors a and b.
// If unsupported metric is requested DistVector returns euclidean ddistance.
// It returns error if the supplied vectors are either nil are different dimension
func Distance(metric string, a, b *mat64.Vector) (float64, error) {
	switch metric {
	case "euclidean":
		return euclideanVec(a, b)
	default:
		return euclideanVec(a, b)
	}
}

// DistanceMx calculates a distance matrix for the given matrix using the given method.
// It returns a hollow symmetric matrix where an item x_ij contains the distance between
// vectors stored in rows i and j. You can request different distance metrics via metric
// parameter. If an unknown metric is supplied Euclidean distance is computed.
func DistanceMx(metric string, matrix *mat64.Dense) (*mat64.Dense, error) {
	switch metric {
	case "euclidean":
		return euclideanMx(matrix)
	default:
		return euclideanMx(matrix)
	}
}

// euclideanVec computes euclidean distances between vectors a and b.
// It returns error if the supplied vectors are either nil are different dimension
func euclideanVec(a, b *mat64.Vector) (float64, error) {
	if a == nil || b == nil {
		return 0.0, fmt.Errorf("Invalid vectors supplied. a: %v, b: %v\n", a, b)
	}
	if a.Len() != b.Len() {
		return 0.0, fmt.Errorf("Incorrect vector dims. a: %d, b: %d\n", a.Len(), b.Len())
	}

	d := mat64.NewVector(a.Len(), nil)
	d.ScaleVec(-1.0, b)
	d.AddVec(a, d)
	d.MulElemVec(d, d)

	return math.Sqrt(mat64.Sum(d)), nil
}

func euclideanMx(in *mat64.Dense) (*mat64.Dense, error) {
	if in == nil {
		return nil, fmt.Errorf("Invalid matrix supplied: %v\n", in)
	}

	rows, _ := in.Dims()
	out := mat64.NewDense(rows, rows, nil)

	for row := 0; row < rows-1; row++ {
		dist := 0.0
		a := in.RowView(row)
		for i := row + 1; i < rows; i++ {
			if i != row {
				b := in.RowView(i)
				dist, _ = euclideanVec(a, b)
				out.Set(row, i, dist)
				out.Set(i, row, dist)
			}
		}
	}

	return out, nil
}
