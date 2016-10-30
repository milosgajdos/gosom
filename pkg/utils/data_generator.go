package utils

import (
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// Creates random data samples clustered in the given number of clusters.
// The samples are located in a hypercube given by the max and min parameters.
// The clusters are hypercubes (not hyperspheres) around randomly picked cluster centres.
// There is the same number of data samples in each cluster (rows/clusters)
// rows - how many data rows to generate
// cols - dimension of data
// clusters - how many clusters
// max,min - maximum and minimum coordinates (applies to all dimensions)
// maxOffset - this is the maximum distance of a sample from its cluster centre in any dimension.
// randSeed - random seed
func CreateClusteredData(rows, cols, clusters int, max, min float64, maxOffset float64, randSeed int64) *mat64.Dense {
	rand.Seed(randSeed)

	data := mat64.NewDense(rows, cols, nil)

	// randomly pick cluster centres
	clusterCentres := make([][]float64, clusters)
	for i := 0; i < clusters; i++ {
		clusterCentres[i] = randVector(max, min, cols)
	}

	for i := 0; i < rows; i++ {
		clusterId := i % clusters
		rv := randVector(maxOffset, -maxOffset, cols)
		cc := make([]float64, cols)
		copy(cc, clusterCentres[clusterId])

		dataPoint := mat64.NewVector(cols, cc)
		rVec := mat64.NewVector(cols, rv)

		dataPoint.AddVec(dataPoint, rVec)
		data.SetRow(i, dataPoint.RawVector().Data)
	}

	return data
}

func randVector(max, min float64, cols int) []float64 {
	v := make([]float64, cols)
	for i := 0; i < cols; i++ {
		v[i] = rand.Float64()*(max-min) + min
	}
	return v
}
