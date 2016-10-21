package som

import (
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/pkg/utils"
)

// CodebookInitFunc defines SOM codebook initialization function
type CodebookInitFunc func(*mat64.Dense, []int) (*mat64.Dense, error)

// CoordsInitFunc defines SOM grid coordinates initialization function
type CoordsInitFunc func(string, []int) (*mat64.Dense, error)

// NeighbFunc defines SOM neighbourhood function
type NeighbFunc func(float64, float64) float64

// Map is a Self Organizing Map (SOM)
type Map struct {
	// codebook is a matrix which contains SOM codebook vectors
	// codebook dimensions: SOM units x data features
	codebook *mat64.Dense
	// unitDist is a symmetric hollow matrix that maps distances between SOM units
	// unitDist dimesions: SOM units x SOM units
	unitDist *mat64.Dense
	// bmus stores codebook row indices of Best Match Units (BMU) over the training
	// bmus will give us an indication of how many clusters are there in the data
	bmus map[int]int
}

// NewMap creates new SOM based on the provided configuration and input data
// NewMap allows you to pass in SOM codebook init function that is used to initialize
// SOM codebook vectors to initial values. If codebook InitFunc is nil, random initialization
// is used. NewMap returns error if the provided configuration is not valid or if the data matrix
// is nil or if the codebook matrix could not be initialized.
func NewMap(c *MapConfig, data *mat64.Dense) (*Map, error) {
	// if input data is empty throw error
	if data == nil {
		return nil, fmt.Errorf("Invalid input data: %v\n", data)
	}
	// validate the map configuration
	if err := validateMapConfig(c); err != nil {
		return nil, err
	}
	// compute the number of map units
	mUnits := utils.IntProduct(c.Dims)
	if mUnits <= 1 {
		return nil, fmt.Errorf("Incorrect map size dimensions: %v\n", c.Dims)
	}
	// initialize codebook
	codebook, err := c.InitFunc(data, c.Dims)
	if err != nil {
		return nil, err
	}
	// grid coordinates matrix
	gridCoords, err := GridCoords(c.UShape, c.Dims)
	if err != nil {
		return nil, err
	}
	// unit distance matrix
	unitDist, err := DistanceMx("euclidean", gridCoords)
	if err != nil {
		return nil, err
	}
	bmus := make(map[int]int)
	// return pointer to new map
	return &Map{
		codebook: codebook,
		unitDist: unitDist,
		bmus:     bmus,
	}, nil
}

// Codebook returns a matrix which contains SOM codebook vectors
func (m Map) Codebook() *mat64.Dense {
	return m.codebook
}

// UnitDist returns a matrix which contains Euclidean distances between SOM units
func (m Map) UnitDist() *mat64.Dense {
	return m.unitDist
}

// BMUs returns a slice which contains indices of Best Match Units (BMUs) of each input vector
func (m Map) BMUs() map[int]int {
	return m.bmus
}

// MarshalTo serializes SOM codebook in a given format.
// At the moment only the native gonum binary format is supported
func (m *Map) MarshalTo(format string, w io.Writer) (int, error) {
	switch format {
	case "gonum":
		return m.codebook.MarshalBinaryTo(w)
	}
	// marshal binary to file path
	return 0, fmt.Errorf("Unsupported format: %s\n", format)
}

// UMatrixOut generates SOM u-matrix in a given format
// At the moment only SVG format is supported
func (m Map) UMatrixOut(format, uShape, title string, w io.Writer) error {
	// TODO: needs some UMatrixSVG modifications first
	return nil
}

// Train runs a SOM training for a given data set and training configuration parameters
// Train modifies the SOM codebook vectors according to the chosen training method.
// If batch method is used, iters is ignored and set to the number of data samples.
// It returns error if the supplied training configuration is invalid
func (m *Map) Train(c *TrainConfig, data *mat64.Dense, iters int) error {
	// number of iterations must be a positive integer
	if iters <= 0 {
		return fmt.Errorf("Invalid number of iterations: %d\n", iters)
	}
	// nil data passed in
	if data == nil {
		return fmt.Errorf("Invalid data supplied: %v\n", data)
	}
	// validate the training configuration
	if err := validateTrainConfig(c); err != nil {
		return err
	}
	// run the training
	switch c.Method {
	case "seq":
		return m.seqTrain(c, data, iters)
	case "batch":
		return m.batchTrain(c, data, iters)
	}

	return nil
}

// seqTrain runs sequential SOM training on a given data set
func (m *Map) seqTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	rows, _ := data.Dims()
	// create random number generator
	rSrc := rand.NewSource(time.Now().UnixNano())
	r := rand.New(rSrc)
	// retrieve Neighbourhood function
	nFn := Neighb[tc.NeighbFn]
	// perform iters number of learning iterations
	for i := 0; i < iters; i++ {
		// pick a random sample from dataset
		sample := data.RowView(r.Intn(rows))
		// no need to check for error here:
		// sample and codebook are not nil and are of the same dimension
		bmu, _ := ClosestVec("euclidean", sample, m.codebook)
		if _, ok := m.bmus[bmu]; !ok {
			m.bmus[bmu] = 1
		}
		// TODO: not thread safe!
		m.bmus[bmu]++
		// no need to check for errors:
		// LRate and Radius are checked by config validation
		lRate, _ := LRate(i, iters, tc.LDecay, tc.LRate)
		radius, _ := Radius(i, iters, tc.RDecay, tc.Radius)
		// pick the bmu distance row
		bmuDists := m.unitDist.RowView(bmu)
		for i := 0; i < bmuDists.Len(); i++ {
			// bmu distance to i-th map unit
			dist := bmuDists.At(i, 0)
			// we are within BMU radius
			if dist < radius {
				cbVec := m.codebook.RowView(i)
				m.seqUpdateCbVec(cbVec, sample, lRate, radius, dist, nFn)
			}
		}
	}

	return nil
}

// seqUpdateCbVec updates vector cbVec given the learning rate l, radius r and distance d
func (m *Map) seqUpdateCbVec(cbVec, vec *mat64.Vector, l, r, d float64, nFn NeighbFunc) {
	// pick codebook vector that should be updated
	diff := mat64.NewVector(cbVec.Len(), nil)
	diff.AddScaledVec(vec, -1.0, cbVec)
	mul := l
	// nFn returns 1 for d == 0; skipping this will save us some CPU time
	if d > 0.0 {
		mul *= nFn(d, r)
	}
	cbVec.AddScaledVec(cbVec, mul, diff)
}

// dataRow holds data vector v, its position in data matrix ix
type dataRow struct {
	vec  *mat64.Vector
	idx  int
	iter int
}

// batchConfig holds batch configuration
type batchConfig struct {
	tc     *TrainConfig
	cbMx   *mat64.Dense
	distMx *mat64.Dense
}

// batchResult holds vector and neighbourhood batch result
type batchResult struct {
	vec  *mat64.Vector
	nghb float64
	idx  int
}

// batchTrain runs batch SOM training on a given data set
func (m *Map) batchTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	cbRows, _ := m.codebook.Dims()
	rows, cols := data.Dims()
	// batchSize set to min(cbRows,rows)
	batchSize := cbRows
	if rows < cbRows {
		batchSize = rows
	}
	// number of worker goroutines
	workers := runtime.NumCPU()
	// number of samples in the batch
	var bound int
	// train for a number of iterations
	for i := 0; i < iters; i++ {
		for j := 0; j < rows; j += batchSize {
			// make data channel a buffered channel
			rowChan := make(chan *dataRow, workers*4)
			// batch results channel
			resChan := make(chan *batchResult, workers)
			// upper bound of batch submatrix
			bound = j + batchSize
			if bound > rows {
				bound = rows
			}
			// batch data matrix: submatrix of data matrix
			batch := data.View(j, 0, bound-j, cols)
			//fmt.Println("from", j, "to", bound-1)
			// goroutine which feeds worker goroutines
			go readDataRows(batch, i, j, rowChan)
			// batchConfig: training config, codebook and unit dist matrix
			bc := &batchConfig{tc: tc, cbMx: m.codebook, distMx: m.unitDist}
			// start worker goroutines
			for j := 0; j < workers; j++ {
				go processRow(resChan, iters, bc, rowChan)
			}
			// collect batch results from all workers
			cbVecs := make([]*mat64.Vector, cbRows)
			cbNghs := make([]float64, cbRows)
			for i := 0; i < workers; i++ {
				res := <-resChan
				if cbVecs[res.idx] != nil {
					cbVecs[res.idx].AddVec(cbVecs[res.idx], res.vec)
				} else {
					cbVecs[res.idx] = res.vec
				}
				cbNghs[res.idx] += res.nghb
			}
			// update codebook vectors
			for i := 0; i < cbRows; i++ {
				if cbVecs[i] != nil {
					cbVecs[i].ScaleVec(1.0/cbNghs[i], cbVecs[i])
					m.codebook.SetRow(i, cbVecs[i].RawVector().Data)
				}
			}
		}
	}

	return nil
}

// readDataRows reads data rows and sends them down rowChan channel
func readDataRows(batch mat64.Matrix, iter, idx int, rowChan chan<- *dataRow) {
	// create batches of data
	rows, _ := batch.Dims()
	// iterate through all batch rows
	for i := 0; i < rows; i++ {
		//fmt.Println("Data mx position: ", i+idx)
		rowChan <- &dataRow{
			iter: iter,
			vec:  (batch.(*mat64.Dense)).RowView(i),
			idx:  i + idx,
		}
	}
	// close channel when done
	close(rowChan)
}

// processRow processes data rows and sends tehm down the results channel
func processRow(res chan<- *batchResult, iters int, bc *batchConfig, rows <-chan *dataRow) {
	// retrieve Neighbourhood function
	neighbFn := Neighb[bc.tc.NeighbFn]
	for row := range rows {
		// find codebook BMU for this data row
		bmu, _ := ClosestVec("euclidean", row.vec, bc.cbMx)
		// calculate radius for this iteration
		radius, _ := Radius(row.iter, iters, bc.tc.RDecay, bc.tc.Radius)
		// pick the BMU's distance row
		bmuDists := bc.distMx.RowView(bmu)
		for i := 0; i < bmuDists.Len(); i++ {
			// bmu distance to i-th map unit
			dist := bmuDists.At(i, 0)
			// when in BMU radius, scale and add to all neighbourhood vecs
			if dist < radius {
				// calculate neighbourhood function
				nghb := neighbFn(dist, radius)
				// allocate new vector
				vec := new(mat64.Vector)
				vec.CloneVec(row.vec)
				vec.ScaleVec(nghb, vec)
				// send batchResult down results channel
				res <- &batchResult{vec: vec, nghb: nghb, idx: i}
			}
		}
	}
}
