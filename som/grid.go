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

// Grid is a SOM grid
type Grid struct {
	// size holds grid dimensions
	size []int
	// ushape holds grid unit shape
	ushape string
	// coords holds grid point coordinates
	coords *mat64.Dense
}

// NewGrid creates new grid and returns it
// It fails with error if the supplied configuration is incorrect
func NewGrid(c *GridConfig) (*Grid, error) {
	// validate the map configuration
	if err := validateGridConfig(c); err != nil {
		return nil, err
	}

	// grid coordinates matrix
	coords, err := GridCoords(c.UShape, c.Size)
	if err != nil {
		return nil, err
	}

	return &Grid{
		size:   c.Size,
		ushape: c.UShape,
		coords: coords,
	}, nil
}

// Size returns a slice that contains Grid dimensions
func (g *Grid) Size() []int {
	return g.size
}

// UShape returns grid unit shape
func (g *Grid) UShape() string {
	return g.ushape
}

// Coords returns a matrix that contains grid coordinates
func (g *Grid) Coords() mat64.Matrix {
	return g.coords
}

// GridSize tries to estimate the best dimensions of map from data matrix and given unit shape.
// It determines the grid size from eigenvectors of input data: the grid dimensions are
// calculated from the ratio of two highest input eigenvalues.
// It returns error if the map dimensions could not be calculated.
func GridSize(data *mat64.Dense, uShape string) ([]int, error) {
	// data matrix can't be nil
	if data == nil {
		return nil, fmt.Errorf("invalid data matrix: %v", data)
	}
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
	// We have more than 2 samples and at least 2D data
	// Calculate eigenvalue ie. SVD singular values
	// Vars() returned here are actually their square values -
	// this does not matter as we are using them to compute their ratios
	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		return nil, fmt.Errorf("Could not determine Principal Components")
	}
	eigVals := pc.Vars(nil)
	// by default we use 1:1 ratio of the map
	ratio := 1.0
	// pick first two components: we only support 2D data maps
	// length check here is redundant, but let's make sure just in case
	if len(eigVals) >= 2 {
		if eigVals[0] != 0 && eigVals[1]*mUnits >= eigVals[0] {
			ratio = math.Sqrt(eigVals[0] / eigVals[1])
		}
	}
	// For hexagon unit shape, the ratio is modified a bit to take it into account
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
// in particular matrix column. The returned matrix has product(dims) number of rows and
// as many columns as the matrix passed in as a parameter.
// It fails with error if the new matrix could not be initialized or if data is nil.
func RandInit(data *mat64.Dense, dims []int) (*mat64.Dense, error) {
	// if nil matrix is passed in, return error
	if data == nil {
		return nil, fmt.Errorf("invalid input matrix: %v", data)
	}
	// dims can't be nil
	if dims == nil {
		return nil, fmt.Errorf("invalid dimensions: %v", dims)
	}
	// dims can't be nil or negative
	for _, dim := range dims {
		if dim <= 0 {
			return nil, fmt.Errorf("Non-Positive dimensions supplied: %v", dims)
		}
	}
	// input matrix dimensions
	_, cols := data.Dims()
	// get max of each column
	max, err := matrix.ColsMax(cols, data)
	if err != nil {
		return nil, err
	}
	// get min of each column
	min, err := matrix.ColsMin(cols, data)
	if err != nil {
		return nil, err
	}
	mUnits := utils.IntProduct(dims)
	// initialize matrix to rand values between 0.0 and 1.0
	codebook, err := matrix.MakeRandom(mUnits, cols, 0.0, 1.0)
	if err != nil {
		return nil, err
	}
	for i := 0; i < cols; i++ {
		col := codebook.ColView(i)
		for j := 0; j < mUnits; j++ {
			val := col.At(j, 0)
			col.SetVec(j, val*(max[i]-min[i])+min[i])
		}
	}
	return codebook, nil
}

// LinInit returns a matrix initialized to values lying in a linear space
// spanned by principal components of data stored in the data matrix passed in as parameter.
// It fails with error if the new matrix could not be initialized or if data is nil.
func LinInit(data *mat64.Dense, dims []int) (*mat64.Dense, error) {
	if err := validateLinInit(data, dims); err != nil {
		return nil, err
	}
	// Adjust map dimensions size to account for 1D cases
	mapDim := len(dims)
	for _, dim := range dims {
		if dim == 1 {
			mapDim--
		}
	}
	// calculate linear space basis
	mapVecs, err := getBaseVecs(data, mapDim)
	if err != nil {
		return nil, err
	}

	_, dataDim := data.Dims()
	mUnits := utils.IntProduct(dims)
	// initialize codebook matrix
	codebook := mat64.NewDense(mUnits, dataDim, nil)
	if dataDim > 1 {
		// calculate mean values of all features in data matrix
		colsMean, err := matrix.ColsMean(dataDim, data)
		if err != nil {
			return nil, err
		}
		// initialize codebok to feature/column means
		for i := 0; i < mUnits; i++ {
			codebook.SetRow(i, colsMean)
		}
		// calculate normalized coordinates
		coords, err := getLinMapCoords(mapDim, dims)
		if err != nil {
			return nil, err
		}
		// generate codebook vectors
		cbRow := make([]float64, dataDim)
		// this can be confusing but mapVecs rows == dataDim == data cols
		mapCol := make([]float64, dataDim)
		for i := 0; i < mUnits; i++ {
			for j := 0; j < mapDim; j++ {
				// grab 1st map eigenvector i.e. base vector
				mat64.Col(mapCol, j, mapVecs)
				floats.Scale(coords.At(i, j), mapCol)
				// grab first codebook row
				mat64.Row(cbRow, i, codebook)
				floats.Add(cbRow, mapCol)
				codebook.SetRow(i, cbRow)
			}
		}
		return codebook, nil
	}
	// calculate 1D option
	min := mat64.Min(data)
	max := mat64.Max(data)
	for i := 0; i < mUnits; i++ {
		val := (float64(i)/(float64(mUnits)-1))*(max-min) + min
		codebook.Set(i, 0, val)
	}

	return codebook, nil
}

// validateLinInit checks whether you can initialize SOM given the provided parameters
// It returns error if at least one of the mandatory conditions fails to be satisfied
func validateLinInit(data *mat64.Dense, dims []int) error {
	// if nil matrix is passed in, return error
	if data == nil {
		return fmt.Errorf("invalid data matrix: %v", data)
	}
	// can't pass in nil dimensions
	if dims == nil {
		return fmt.Errorf("Incorrect dimensions: %v", dims)
	}
	// Check if any of the supplied dimensions are non-negative
	for _, dim := range dims {
		if dim <= 0 {
			return fmt.Errorf("Non-Positive dimensions supplied: %v", dims)
		}
	}
	// Linear initialization requires at least 2 samples
	samples, _ := data.Dims()
	if samples < 2 {
		return fmt.Errorf("Insufficient number of samples: %d", samples)
	}
	return nil
}

// getBaseVecs calculates linear space base vectors from the provided data
// It returns a matrix that contains the linear space base vectors.
// It fails with error if the principal components could not be found
func getBaseVecs(data *mat64.Dense, mapDim int) (*mat64.Dense, error) {
	// mapVecs is a matrix that holds the map linear base vectors
	baseVecs := new(mat64.Dense)
	// If both data dimension and requested map dimensions are >1 do PCA
	// In other words we only do PCA if the map has at least 2 dimensions
	// and if the real map dimensions are at most the same as data dimensions
	samples, dataDim := data.Dims()
	if dataDim > 1 && (dataDim >= mapDim) {
		// calculate Principal components of the input data
		var pc stat.PC
		ok := pc.PrincipalComponents(data, nil)
		if !ok {
			return nil, fmt.Errorf("Could not determine Principal Components")
		}
		// principal components and their eigen values
		vecs := pc.Vectors(nil)
		vals := pc.Vars(nil)
		// normalize the eigenvectors
		for i := 0; i < mapDim; i++ {
			vec := vecs.ColView(i)
			vec.ScaleVec(math.Sqrt(vals[i])/mat64.Norm(vec, 2), vec)
		}
		//fb = mat64.Formatted(vecs, mat64.Prefix("    "))
		//fmt.Printf("NORMALIZED PCA vecs:a = %v", fb)
		// pick first m eigenvectors
		baseVecs.Clone(vecs.View(0, 0, dataDim, mapDim))
	} else {
		// we have only 1D data i.e. 1 column - let's get standard deviation
		col := make([]float64, samples)
		stdev := stat.StdDev(mat64.Col(col, 0, data), nil)
		baseVecs.Grow(1, 1)
		baseVecs.Set(0, 0, stdev)
	}
	return baseVecs, nil
}

// getLinMapCoords calculates map coordinates and normalizes them to unit values
// It returns error if it can't calculate coordinates
func getLinMapCoords(mapDim int, dims []int) (*mat64.Dense, error) {
	// calculate unit coordinates
	coords, err := GridCoords("rectangle", dims)
	if err != nil {
		return nil, err
	}
	// swap x and y coordinates
	mUnits := utils.IntProduct(dims)
	x := make([]float64, mUnits)
	y := make([]float64, mUnits)
	mat64.Col(x, 0, coords)
	mat64.Col(y, 1, coords)
	coords.SetCol(0, y)
	coords.SetCol(1, x)
	// normalize coordinates to unit values
	c := make([]float64, mUnits)
	for i := 0; i < mapDim; i++ {
		c = mat64.Col(c, i, coords)
		max := floats.Max(c)
		min := floats.Min(c)
		if max > min {
			floats.AddConst(-min, c)
			floats.Scale(1.0/(max-min), c)
		} else {
			for j := 0; j < len(c); j++ {
				c[j] = 0.5
			}
		}
		coords.SetCol(i, c)
	}
	m, err := matrix.AddConst(-0.5, coords)
	if err != nil {
		return nil, err
	}
	m.Scale(2, m)
	return m, nil
}

// GridCoords returns a matrix which contains coordinates of all SOM units stored row by row.
// dims specify the size of the Grid, so the returned matrix has as many rows as is the
// product of the numbers stored in dims slice and as many columns as is the length of dims slice.
// GridCoords fails with error if the requested unit shape is unsupported or if the incorrect
// dimensions are supplied: dims slice can't be nil nor can its length be bigger than 3
func GridCoords(uShape string, dims []int) (*mat64.Dense, error) {
	// validate passed in parameter
	if err := validateGridCoords(uShape, dims); err != nil {
		return nil, err
	}
	mDims := len(dims)
	// turn [n] -> [n,1]
	if mDims == 1 {
		dims = append(dims, 1)
	}
	// We need coordinates tuples to populate coords matrix
	// cumprod will give us counts/length of the tuples of same numbers in each sequence
	// dims will give us the the upper bound of the tuple sequence
	// i.e. cumprod = 2, dim = 2 means we will have [2, 2] tubple in the sequence
	counts := utils.IntCumProduct(dims)
	counts = append([]int{1}, counts...)
	// init grid coordinates matrix
	mUnits := utils.IntProduct(dims)
	coords := mat64.NewDense(mUnits, mDims, nil)
	for i := 0; i < mDims; i++ {
		seq := makeSeq(mUnits/counts[i+1], dims[i], counts[i])
		coords.SetCol(i, seq)
	}
	// retrieve x and y coords
	x := mat64.Col(make([]float64, mUnits), 0, coords)
	y := mat64.Col(make([]float64, mUnits), 1, coords)
	// swaps x and y coordinates: ij notation to xy
	if mDims >= 2 {
		coords.SetCol(1, x)
		coords.SetCol(0, y)
	}
	// This will offset x-coordinates of every other unit by 0.5.
	// This will make distances of a unit to all its six neighbors equal
	if strings.EqualFold(uShape, "hexagon") {
		x = mat64.Col(x, 0, coords)
		y = mat64.Col(y, 1, coords)
		// dims[1] was y-dim, before we swapped it for x-dim
		xDim := dims[1]
		repCount := mUnits / xDim
		for i := 0; i < xDim; i++ {
			for j := repCount*i + 1; j < repCount*(i+1); j += 2 {
				x[j] += 0.5
			}
		}
		// y-coordinates are multiplied by sqrt(0.75)
		for i := 0; i < mUnits; i++ {
			y[i] *= math.Sqrt(0.75)
		}
		coords.SetCol(0, x)
		coords.SetCol(1, y)
	}
	return coords, nil
}

// validate gridCoords validates whether you can initialize SOM unit coordinates
// given the provided parameters. It returns error if the validation fails
func validateGridCoords(uShape string, dims []int) error {
	// unsupported SOM unit shape
	if _, ok := uShapes[uShape]; !ok {
		return fmt.Errorf("unsupported unit shape: %s", uShape)
	}
	// map dims can't be nil
	if dims == nil {
		return fmt.Errorf("invalid dimensions supplied: %v", dims)
	}
	// check if the dimensions are postive numbers
	for _, dim := range dims {
		if dim <= 0 {
			return fmt.Errorf("Non-Positive dimensions supplied: %v", dims)
		}
	}
	// map dims can't be longer than 3
	mDims := len(dims)
	if mDims > 3 {
		return fmt.Errorf("unsupported dimensions requested: %d", mDims)
	}
	// can't use hexagon with dims > 2
	if strings.EqualFold(uShape, "hexagon") {
		if mDims > 2 {
			return fmt.Errorf("Exceeded allowed hexagon dims: %d", mDims)
		}
	}
	return nil
}

// makeSeq makes a sequence of numbers and returns it in a slice
// The length of the returned slice is equal to the product of all function parameters:
// reps specifies number of sequence repetitions
// bound specifies an upper bound of sequence
// count specifies number of times the number in a sequence should be repeated
func makeSeq(reps, bound, count int) []float64 {
	seq := make([]float64, count*bound*reps)
	// # repeats of the same sequence
	idx := 0
	for r := 0; r < reps; r++ {
		// which numbers should be in the sequence
		for i := 0; i < bound; i++ {
			// how many times each number repeats
			for j := 0; j < count; j++ {
				seq[idx] = float64(i)
				idx++
			}
		}
	}
	return seq
}
