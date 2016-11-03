package som

import (
	"fmt"
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

// QuantError computes SOM quantization error for the supplied data set and codebook
// It returns the quantization error or fails with error if either data or codebook are nil
// or the distance between the vectors could not be calculated.
// When the error is returned, quantization error is set to -1.0
func QuantError(data, codebook *mat64.Dense) (float64, error) {
	// data can't be nil
	if data == nil {
		return -1.0, fmt.Errorf("Invalid data supplied: %v\n", data)
	}
	// codebook can't be nil
	if codebook == nil {
		return -1.0, fmt.Errorf("Invalid codebook supplied: %v\n", codebook)
	}
	var qErr float64
	metric := "euclidean"
	rows, _ := data.Dims()
	for i := 0; i < rows; i++ {
		bmuIdx, err := ClosestVec(metric, data.RowView(i), codebook)
		if err != nil {
			return -1.0, err
		}
		// get the BMU distance -- no need to check for errors here
		d, err := Distance(metric, data.RowView(i), codebook.RowView(bmuIdx))
		if err != nil {
			return -1.0, err
		}
		qErr += d
	}
	// compute the average distance
	qErr = qErr / float64(rows)

	return qErr, nil
}

// TopoProduct calculates topographic product for given codebook and grid.
// TopoProduct returns error if either codebook or grid are nil or if number of codebook rows
// is not the same as the number of grid rows. If any two codebooks turn out to be the same
// TopoProduct returns +Inf - this can happen when map is trained using batch algorithm.
func TopoProduct(codebook, grid *mat64.Dense) (float64, error) {
	// codebook can't be nil
	if codebook == nil {
		return 0.0, fmt.Errorf("Invalid codebook supplied: %v\n", codebook)
	}
	// grid can't be nil
	if grid == nil {
		return 0.0, fmt.Errorf("Invalid grid supplied: %v\n", grid)
	}
	// if grid and codebook don't match, throw error
	gRows, _ := grid.Dims()
	cRows, _ := codebook.Dims()
	if gRows != cRows {
		return 0.0, fmt.Errorf("Grid and codebook dimension mismatch\n")
	}
	// unit and codebook distance matrices -- no need to check for error here
	uDistMx, _ := DistanceMx("euclidean", grid)
	cDistMx, _ := DistanceMx("euclidean", codebook)
	// tp is topographic product
	var tp float64
	// loop through all neurons
	for i := 0; i < gRows; i++ {
		// retrieve unit distance slice and sort it
		uSlice := newFloat64Slice(uDistMx.RowView(i).RawVector().Data...)
		sort.Sort(uSlice)
		uKNeighb := uSlice.index[1:]
		// retrieve codebook distance slice and sort it
		cSlice := newFloat64Slice(cDistMx.RowView(i).RawVector().Data...)
		sort.Sort(cSlice)
		cKNeighb := cSlice.index[1:]
		// topgraphic product and partial distortion products
		p1, p2, p3 := 1.0, 1.0, 1.0
		for j := 0; j < cRows-1; j++ {
			// lattice_space / codebook_space
			q1 := cDistMx.At(i, uKNeighb[j]) / cDistMx.At(i, cKNeighb[j])
			q2 := uDistMx.At(i, uKNeighb[j]) / uDistMx.At(i, cKNeighb[j])
			// if 2 codebooks/map units are the same, return +Inf
			if math.IsNaN(q1) || math.IsNaN(q2) {
				return math.Inf(1), nil
			}
			// calculate P1, P2, P3
			p1 *= q1
			p2 *= q2
			p3 *= q1 * q2
			// the actual P3 has to be square rooted
			tp += math.Log(math.Pow(p3, 1/float64(2*(j+1))))
			//fmt.Println("P1", math.Pow(p1, 1/float64((j+1))))
			//fmt.Println("P2", math.Pow(p2, 1/float64((j+1))))
		}
	}

	return tp / float64(cRows*(cRows-1)), nil
}

// TopoError calculate topographice error for given data set, codebook and grid and returns it
// It returns error if either data, codebook or grid are nil or if their dimensions are mismatched.
func TopoError(data, codebook, grid *mat64.Dense) (float64, error) {
	// data can't be nil
	if data == nil {
		return -1.0, fmt.Errorf("Invalid data supplied: %v\n", data)
	}
	// codebook can't be nil
	if codebook == nil {
		return -1.0, fmt.Errorf("Invalid codebook supplied: %v\n", codebook)
	}
	// grid can't be nil
	if grid == nil {
		return -1.0, fmt.Errorf("Invalid grid supplied: %v\n", grid)
	}
	// unit distance matrix -- no need to check for error
	uDistMx, _ := DistanceMx("euclidean", grid)
	var te float64
	// iterate through all data samples
	rows, _ := data.Dims()
	for i := 0; i < rows; i++ {
		closest, err := ClosestNVec("euclidean", 2, data.RowView(i), codebook)
		if err != nil {
			return -1.0, err
		}
		// If the 2 BMUS are next to each other on lattice increment te.
		// 1.01*math.Sqrt(2) accounts for voronoi cell neighbourhood.
		// This makes the diagonal lattice units to be considered neighb.
		uDistVec := uDistMx.RowView(closest[0])
		if uDistVec.At(closest[1], 0) >= 1.01*math.Sqrt(2) {
			te++
		}
	}

	return te / float64(rows), nil
}
