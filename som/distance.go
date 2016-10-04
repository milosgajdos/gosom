package som

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func DistanceMatrix(method string, matrix *mat64.Dense) *mat64.Dense {
	switch method {
	case "euclidean":
		return euclidean(matrix)
	default:
		return euclidean(matrix)
	}
}

func euclidean(in *mat64.Dense) *mat64.Dense {
	rows, cols := in.Dims()
	out := mat64.NewDense(rows, rows, nil)

	for row := 0; row < rows; row++ {
		for i := 0; i < rows; i++ {
			dist := 0.0
			if i != row {
				for j := 0; j < cols; j++ {
					dist += math.Pow(in.At(row, j)-in.At(i, j), 2.0)
				}
				dist = math.Sqrt(dist)
			}
			out.Set(row, i, dist)
		}
	}

	return out
}
