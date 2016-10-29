package som

import (
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"sync"
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
	// bmus stores indices of Best Match Units (BMU) for each input data
	bmus []int
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
	rows, _ := data.Dims()
	bmus := make([]int, rows)
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
func (m Map) BMUs() []int {
	return m.bmus
}

// MarshalTo serializes SOM codebook in a given format to writer w.
// At the moment only the native gonum binary format is supported.
// It returns the number of bytes written to w or fails with error.
func (m *Map) MarshalTo(format string, w io.Writer) (int, error) {
	switch format {
	case "gonum":
		return m.codebook.MarshalBinaryTo(w)
	}
	// marshal binary to file path
	return 0, fmt.Errorf("Unsupported format: %s\n", format)
}

// UMatrixOut generates SOM u-matrix in a given format and writes the output to w.
// At the moment only SVG format is supported. It fails with error if the write to w fails.
func (m Map) UMatrixOut(format, title string, w io.Writer) error {
	// TODO: needs some UMatrixSVG modifications first
	return nil
}

// Train runs a SOM training for a given data set and training configuration parameters.
// It modifies the map codebook vectors based on the chosen training algorithm.
// It returns error if the supplied training configuration is invalid or training fails
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

// seqTrain runs sequential SOM training algorithm on a given data set
func (m *Map) seqTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	rows, _ := data.Dims()
	// create random number generator
	rSrc := rand.NewSource(time.Now().UnixNano())
	r := rand.New(rSrc)
	// retrieve Neighbourhood function
	neighbFn := Neighb[tc.NeighbFn]
	// perform iters number of learning iterations
	for i := 0; i < iters; i++ {
		// pick a random sample from dataset
		sample := data.RowView(r.Intn(rows))
		// no need to check for error here:
		// sample and codebook are not nil and have the same dimension
		bmu, _ := ClosestVec("euclidean", sample, m.codebook)
		// no need to check for errors:
		// LRate and Radius are checked by config validation
		lRate, _ := LRate(i, iters, tc.LDecay, tc.LRate)
		radius, _ := Radius(i, iters, tc.RDecay, tc.Radius)
		// pick the bmu unit distance row
		bmuDists := m.unitDist.RowView(bmu)
		// find units which are within the radius
		for i := 0; i < bmuDists.Len(); i++ {
			// bmu distance to i-th map unit
			dist := bmuDists.At(i, 0)
			// we are within BMU radius
			if dist < radius {
				// update particular codebook vector
				m.seqUpdateCbVec(i, sample, lRate, radius, dist, neighbFn)
			}
		}
	}

	return nil
}

// seqUpdateCbVec updates codebook vector on row cbIdx given the learning rate l,
// radius r, distance d and neihgbourhood function nFn
func (m *Map) seqUpdateCbVec(cbIdx int, vec *mat64.Vector, l, r, d float64, nFn NeighbFunc) {
	// pick codebook vector that should be updated
	cbVec := m.codebook.RowView(cbIdx)
	// update codebook vector according to the algorithm
	diff := mat64.NewVector(cbVec.Len(), nil)
	diff.AddScaledVec(vec, -1.0, cbVec)
	mul := l
	// nFn returns 1 for d == 0; skipping this case will save us some CPU time
	if d > 0.0 {
		mul *= nFn(d, r)
	}
	cbVec.AddScaledVec(cbVec, mul, diff)
}

// batchConfig holds batch training configuration
type batchConfig struct {
	// tc is SOM training configuration
	tc *TrainConfig
	// iters is a number of batch iterations
	iters int
}

// batchResult holds result of batch algorithm for a particular data input
// It holds scaled data vector, neighbourhood of its BMU and index of codebook vector
type batchResult struct {
	// vec is a scaled data vector
	vec *mat64.Vector
	// nghb is vec BMU neighbourhood
	nghb float64
	// idx is an index of codebook vector to update
	idx int
}

// batchTrain runs batch SOM training on a given data set
func (m *Map) batchTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	cbRows, _ := m.codebook.Dims()
	rows, _ := data.Dims()
	// bSize set to min(cbRows,rows)
	bSize := cbRows
	if rows < cbRows {
		bSize = rows
	}
	// number of batch iterations in one training iteration
	bIters := rows / bSize
	if rows%bSize != 0 {
		bIters++
	}
	// batchConfig holds training config and number of iterations
	bc := &batchConfig{
		tc:    tc,
		iters: iters * bIters,
	}
	// number of worker goroutines
	workers := runtime.NumCPU()
	// evenly distribute batch work between workers
	workerBatch := bSize / workers
	if workerBatch == 0 {
		workerBatch = 1
	}
	// set the first batch iteration to 0
	iter := 0
	// train for a number of iterations
	for i := 0; i < iters; i++ {
		count := workerBatch
		// Iterate over the input data in batches of size bSize
		for j := 0; j < rows; j += bSize {
			// create batch results channel
			results := make(chan *batchResult, workers*4)
			wg := &sync.WaitGroup{}
			// start worker goroutines
			for k := 0; k < workers; k++ {
				// from is data matrix row pointer
				from := j + k*workerBatch
				// last worker will work through the batch reminder
				if k == workers-1 {
					count += bSize % workers
				}
				// if we go over the number of rows adjust bSamples
				if from+count > rows {
					count = rows - from
				}
				wg.Add(1)
				go m.processBatch(results, wg, bc, data, from, count, iter)
			}
			// wait for workers to finish and close the result channel
			go func() {
				wg.Wait()
				close(results)
			}()
			// collect batch results from all workers
			cbVecs := make([]*mat64.Vector, cbRows)
			nghbs := make([]float64, cbRows)
			for result := range results {
				if cbVecs[result.idx] != nil {
					cbVecs[result.idx].AddVec(cbVecs[result.idx], result.vec)
				} else {
					cbVecs[result.idx] = result.vec
				}
				nghbs[result.idx] += result.nghb
			}
			// update codebook vectors
			for k := 0; k < cbRows; k++ {
				if cbVecs[k] != nil {
					cbVecs[k].ScaleVec(1.0/nghbs[k], cbVecs[k])
					m.codebook.SetRow(k, cbVecs[k].RawVector().Data)
				}
			}
			iter++
		}
	}

	return nil
}

// processRow processes data rows and sends tehm down the results channel
func (m Map) processBatch(res chan<- *batchResult, wg *sync.WaitGroup,
	bc *batchConfig, data *mat64.Dense, from, count, iter int) {
	// retrieve Neighbourhood function
	neighbFn := Neighb[bc.tc.NeighbFn]
	for i := from; i < count+from; i++ {
		row := data.RowView(i)
		// find codebook BMU for this data row
		bmu, _ := ClosestVec("euclidean", row, m.codebook)
		// calculate radius for this iteration
		radius, _ := Radius(iter, bc.iters, bc.tc.RDecay, bc.tc.Radius)
		// pick the BMU's distance row
		bmuDists := m.unitDist.RowView(bmu)
		for j := 0; j < bmuDists.Len(); j++ {
			// bmu distance to i-th map unit
			dist := bmuDists.At(j, 0)
			// when in BMU radius, scale and add to all neighbourhood vecs
			if dist < radius {
				// calculate neighbourhood function
				nghb := neighbFn(dist, radius)
				// allocate new vector and copy row data to it
				vec := new(mat64.Vector)
				vec.CloneVec(row)
				vec.ScaleVec(nghb, vec)
				// send batchResult down results channel
				res <- &batchResult{vec: vec, nghb: nghb, idx: j}
			}
		}
	}
	wg.Done()
}
