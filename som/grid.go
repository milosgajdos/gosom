package som

import (
	"fmt"
	"math"
	"strings"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/milosgajdos83/gosom/pkg/matrix"
	"github.com/milosgajdos83/gosom/pkg/utils"
)

// GridDims tries to estimate the best dimensions of map from data matrix and given unit shape.
// It determines the grid size from eigenvectors of input data: the grid dimensions are
// calculated from the ratio of two highest input eigenvalues.
// It returns error if the map dimensions could not be calculated.
func GridDims(data *mat64.Dense, uShape string) ([]int, error) {
	dataLen, dataDim := data.Dims()
	// this is a simple heuristic - you can pick the scale > 5
	mUnits := math.Ceil(5 * math.Sqrt(float64(dataLen)))
	// if the data is 1D - we return [1 x mUnits] map dimensions
	if dataDim == 1 && dataLen > 1 {
		return []int{1, int(mUnits)}, nil
	}
	// Not enough data to calculate eigenvectors
	// We will use heuristic: number of mUnits = square area of SOM
	if dataLen < 2 {
		gDim := math.Sqrt(mUnits)
		return []int{int(gDim), int(gDim)}, nil
	}
	// We have more than 2 samples and more than 1D data
	// Calculate eigenvalue ie. SVD singular values
	_, eigVals, ok := stat.PrincipalComponents(data, nil)
	if !ok {
		return nil, fmt.Errorf("Could not determine Principal Components")
	}
	// by default we use 1:1 ratio of the map
	ratio := 1.0
	// pick first two components: we only support 2D data maps
	// length check here is redundant, but let's make sure just in case
	if len(eigVals) >= 2 {
		if eigVals[0] != 0 && eigVals[1]*mUnits >= eigVals[0] {
			ratio = math.Sqrt(eigVals[0] / eigVals[1])
		}
	}
	// If the unit shape is hexagon, the ratio is modified a bit to take it into account
	// Remember when using hexagon we don't get rectangle so the area != dimA * dimB
	tmpDim := math.Sqrt(mUnits / ratio)
	if strings.EqualFold(uShape, "hexagon") {
		tmpDim = math.Sqrt(mUnits / ratio * math.Sqrt(0.75))
	}
	yDim := int(floats.Min([]float64{mUnits, tmpDim}))
	xDim := int(mUnits / float64(yDim))
	// Return map dimensions
	return []int{xDim, yDim}, nil
}

// RandInit returns a matrix initialized to uniformly distributed random values
// in each column in range between [max, min] where max and min are maximum and minmum values
// in particular matrix column. The returned matrix has mUnits number of rows and
// as many columns as the matrix passed in as a parameter.
// It fails with error if the new matrix could not be initialized or if inMx is nil.
func RandInit(inMx *mat64.Dense, mUnits int, gridDims []int) (*mat64.Dense, error) {
	// if nil matrix is passed in, return error
	if inMx == nil {
		return nil, fmt.Errorf("Invalid input matrix: %v\n", inMx)
	}
	// input matrix dimensions
	_, cols := inMx.Dims()
	// get max of each column
	max, err := matrix.ColsMax(cols, inMx)
	if err != nil {
		return nil, err
	}
	// get min of each column
	min, err := matrix.ColsMin(cols, inMx)
	if err != nil {
		return nil, err
	}
	// initialize matrix to rand values between 0.0 and 1.0
	outMx, err := matrix.MakeRandom(mUnits, cols, 0.0, 1.0)
	if err != nil {
		return nil, err
	}
	for i := 0; i < cols; i++ {
		col := outMx.ColView(i)
		for j := 0; j < mUnits; j++ {
			val := col.At(j, 0)
			col.SetVec(j, val*(max[i]-min[i])+min[i])
		}
	}
	return outMx, nil
}

// LinInit returns a matrix initialized to values lying in a linear space
// spanned by the top principal components of the values stored in the matrix passed in
// as a parameter. The returned matrix has mUnits number of rows and
// as many columns as the input matrix passed in as a parameter
// It fails with error if the new matrix could not be initialized or if inMx is nil.
func LinInit(inMx *mat64.Dense, mUnits int, gridDims []int) (*mat64.Dense, error) {
	// if nil matrix is passed in, return error
	if inMx == nil {
		return nil, fmt.Errorf("Invalid input matrix: %v\n", inMx)
	}
	r, _ := inMx.Dims()
	// Linear initialization requires at least 2 samples
	if r < 2 {
		return nil, fmt.Errorf("Insufficient number of samples: %d\n", r)
	}
	return nil, nil
}

// PlaneGridCoords returns a matrix that contains coordinates of SOM units
// stored by row. Planar coordinates naturally imply two columns: x and y dimensions
// It fails with error if the requested unit shape is unsupported or if the incorrect
// dimensions are supplied: sims slice can't be nil nor can its length be bigger than 2
func PlaneGridCoords(uShape string, dims []int) (*mat64.Dense, error) {
	// unsupported SOM unit shape
	if _, ok := UShape[uShape]; !ok {
		return nil, fmt.Errorf("Unsupported unit shape: %s\n", uShape)
	}
	// map dims can't be nil or bigger than 2
	mDims := len(dims)
	if dims == nil || mDims != 2 {
		return nil, fmt.Errorf("Invalid plane dimensions supplied: %v\n", dims)
	}
	mUnits := utils.IntProduct(dims)
	coords := mat64.NewDense(mUnits, mDims, nil)
	// we need coordinates pairs [x,y] to populate coords matrix
	xDim, yDim := dims[0], dims[1]
	xCoords := makeXCoords(mUnits, xDim)
	yCoords := makeYCoords(mUnits, yDim)
	// if hexagon this will make distances of a unit to all its six neighbors equal
	if strings.EqualFold(uShape, "hexagon") {
		// offset x-coordinates of every other row by 0.5
		seq := mUnits / xDim
		for i := 0; i < seq; i++ {
			for j := 1 + i*xDim; j < (i+1)*xDim; j += 2 {
				xCoords[j] += 0.5
			}
		}
		// y-coordinates are multiplied by sqrt(0.75)
		for i := 0; i < mUnits; i++ {
			yCoords[i] *= math.Sqrt(0.75)
		}
	}
	// populate coords matrix with coordinates
	coords.SetCol(0, xCoords)
	coords.SetCol(1, yCoords)
	return coords, nil
}

// makeXCoords generates X-coords and returns them in a slice
// X coordinats look like this: 0 0 0 1 1 1 2 2 2
func makeXCoords(mUnits, dim int) []float64 {
	// allocate coords array
	x := make([]float64, mUnits)
	val := 0.0
	for i := 0; i < mUnits; i++ {
		if i != 0 && i%dim == 0 {
			val += 1.0
		}
		x[i] = val
	}
	return x
}

// makeYCoords generates Y-coords and returns them in a slice
// Y coordinats look like this: 0 1 2 0 1 2 0 1 2
func makeYCoords(mUnits, dim int) []float64 {
	y := make([]float64, mUnits)
	// generate Y-coords
	val := 0.0
	for i := 0; i < mUnits; i++ {
		if i != 0 && i%dim == 0 {
			val = 0.0
		}
		y[i] = val
		val += 1.0
	}
	return y
}
