package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(1000, 200, 20, 100.0, 0.0, 1.0, 10)

	TIME := time.Now()
	coords, mUnits, coordsDims := runSom(data)
	printTimer(TIME)
	som.CreateSVG(coords, mUnits, coordsDims, "hexagon", "Done", false, "umatrix.html")
}

func runSom(data *mat64.Dense) (*mat64.Dense, *mat64.Dense, []int) {
	TIME := time.Now()

	totalIterations, _ := data.Dims()
	// SOM configuration
	mConfig := &som.MapConfig{
		Dims:     []int{43, 36},
		InitFunc: som.RandInit,
		Grid:     "planar",
		UShape:   "hexagon",
	}
	// create new SOM map
	smap, _ := som.NewMap(mConfig, data)

	tConfig := &som.TrainConfig{
		Method:   "seq",
		Radius:   10.0,
		RDecay:   "exp",
		NeighbFn: "gaussian",
		LRate:    0.5,
		LDecay:   "exp",
	}

	smap.Train(tConfig, data, totalIterations)
	printTimer(TIME)

	coords, _ := som.GridCoords(mConfig.UShape, mConfig.Dims)
	return coords, smap.Codebook(), mConfig.Dims
}

func data(rows, cols, clusters int, max, min float64, vari float64, randSeed int64) *mat64.Dense {
	rand.Seed(randSeed)

	data := mat64.NewDense(rows, cols, nil)

	clusterCentres := make([][]float64, clusters)
	for i := 0; i < clusters; i++ {
		clusterCentres[i] = randVector(max, min, cols)
		//fmt.Printf("Cluster %d = %v\n", i, clusterCentres[i])
	}

	for i := 0; i < rows; i++ {
		clusterId := i % clusters
		rv := randVector(vari, -vari, cols)
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

func printMatrix(matrix *mat64.Dense) {
	rows, _ := matrix.Dims()
	for i := 0; i < rows; i++ {
		fmt.Println(matrix.RawRowView(i))
	}
}

func printTimer(TIME time.Time) {
	elapsed := float64((time.Now().UnixNano() - TIME.UnixNano()))
	units := []string{"ns", "us", "ms", "s"}
	for i, unit := range units {
		if elapsed > 1000.0 && i < (len(units)-1) {
			elapsed /= 1000.0
		} else {
			fmt.Printf("%f%s\n", elapsed, unit)
			break
		}
	}
}
