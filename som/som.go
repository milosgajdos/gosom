package som

import (
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Map is a Self Organizing Map (SOM)
type Map struct {
	// codebook is a matrix which contains SOM codebook vectors
	// codebook dimensions: SOM units x data features
	codebook *mat.Dense
	// grid is a matrix which contains SOM unit coordinages
	// grid dimensions depend on chosen configuration
	grid *Grid
}

// NewMap creates new SOM based on the provided configuration.
// It creates a map grid and initializes codebook vectors using the provided configuration parameter.
// NewMap returns error if the provided configuration is not valid or if the data matrix is nil or
// if the codebook matrix could not be initialized.
// TODO: Avoid passing in data matrix when creating new map
func NewMap(c *MapConfig, data *mat.Dense) (*Map, error) {
	// if input data is empty throw error
	if data == nil {
		return nil, fmt.Errorf("invalid input data: %v", data)
	}
	// validate codebook config
	if err := validateCbConfig(c.Cb); err != nil {
		return nil, err
	}
	// initialize codebook
	codebook, err := c.Cb.InitFunc(data, c.Grid.Size)
	if err != nil {
		return nil, err
	}
	// make new grid
	grid, err := NewGrid(c.Grid)
	if err != nil {
		return nil, err
	}
	// return pointer to new map
	return &Map{
		codebook: codebook,
		grid:     grid,
	}, nil
}

// Codebook returns a matrix which contains SOM codebook vectors
func (m Map) Codebook() mat.Matrix {
	return m.codebook
}

// Grid returns SOM grid
func (m Map) Grid() *Grid {
	return m.grid
}

// UnitDist returns a matrix which contains Euclidean distances between SOM units
func (m Map) UnitDist() (*mat.Dense, error) {
	return DistanceMx("euclidean", m.grid.coords)
}

// BMUs returns a slice which contains indices of Best Match Unit vectors to the map
// codebook for each vector stored in data rows.
// It returns error if the data dimension and map codebook dimensions are not the same.
func (m Map) BMUs(data *mat.Dense) ([]int, error) {
	return BMUs(data, m.codebook)
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
	return 0, fmt.Errorf("unsupported format: %s", format)
}

// UMatrix generates SOM u-matrix in a given format and writes the output to w.
// At the moment only SVG format is supported. It fails with error if the write to w fails.
func (m Map) UMatrix(w io.Writer, data *mat.Dense, classMap map[int]int, format, title string) error {
	switch format {
	case "svg":
		{
			// map that contains most frequent BMU class of all of its classes
			bmuClassMap := make(map[int]int)
			// only do this if we supply data class map
			if len(classMap) > 0 {
				bmuClasses, err := m.mapBMUclasses(data, classMap)
				if err != nil {
					return err
				}

				// find the most frequent class for each codebook vector
				for cbi, class := range bmuClasses {
					sort.Ints(class)
					count := 1
					currentIndex := 0
					for i := 1; i < len(class); i++ {
						if class[i] == class[currentIndex] {
							count++
						} else {
							count--
						}
						if count == 0 {
							currentIndex = i
							count = 1
						}
					}
					bmuClassMap[cbi] = class[currentIndex]
				}
			}

			return UMatrixSVG(m.codebook, m.grid.size, m.grid.ushape, title, w, bmuClassMap)
		}
	}

	return fmt.Errorf("invalid format %s", format)
}

// mapBMUclasses returns a map which contains a list of classes to which this BMUs input samples are members of
// We go through all data samples and add their classes to the list of classes of their respective BMUs.
func (m Map) mapBMUclasses(data *mat.Dense, classes map[int]int) (map[int][]int, error) {
	rows, _ := data.Dims()
	// map of all classes for each BMU
	bmuClasses := make(map[int][]int)
	for row := 0; row < rows; row++ {
		// find BMU
		cbi, err := ClosestVec("euclidean", data.RawRowView(row), m.codebook)
		if err != nil {
			return nil, err
		}
		// lookup input sample class
		class, ok := classes[row]
		if ok {
			classList, ok := bmuClasses[cbi]
			if !ok {
				classList = []int{}
			}
			// append class to BMU list of classes
			classList = append(classList, class)
			bmuClasses[cbi] = classList
		}
	}

	return bmuClasses, nil
}

// Train runs a SOM training for a given data set and training configuration parameters.
// It modifies the map codebook vectors based on the chosen training algorithm.
// It returns error if the supplied training configuration is invalid or training fails
func (m *Map) Train(c *TrainConfig, data *mat.Dense, iters int) error {
	// number of iterations must be a positive integer
	if iters <= 0 {
		return fmt.Errorf("invalid number of iterations: %d", iters)
	}
	// nil data passed in
	if data == nil {
		return fmt.Errorf("invalid data supplied: %v", data)
	}
	// validate the training configuration
	if err := validateTrainConfig(c); err != nil {
		return err
	}
	// run the training
	switch c.Algorithm {
	case "seq":
		return m.seqTrain(c, data, iters)
	case "batch":
		return m.batchTrain(c, data, iters)
	}

	return nil
}

// QuantError computes SOM quantization error for the supplied data set
// It returns the quantization error or fails with error if the passed in data is nil
// or the distance betweent vectors could not be calculated.
// When the error is returned, quantization error is set to -1.0.
func (m Map) QuantError(data *mat.Dense) (float64, error) {
	return QuantError(data, m.codebook)
}

// TopoProduct computes SOM topographic product
// It returns a single number or fails with error if the product could not be computed
func (m Map) TopoProduct() (float64, error) {
	return TopoProduct(m.codebook, m.grid.coords)
}

// TopoError computes SOM topographic error for a given data set.
// It returns a single number or fails with error if the error could not be computed
func (m Map) TopoError(data *mat.Dense) (float64, error) {
	return TopoError(data, m.codebook, m.grid.coords)
}

// seqTrain runs sequential SOM training algorithm on a given data set
func (m *Map) seqTrain(tc *TrainConfig, data *mat.Dense, iters int) error {
	rows, _ := data.Dims()
	// create random number generator
	rSrc := rand.NewSource(time.Now().UnixNano())
	r := rand.New(rSrc)
	// calculate unit distances
	unitDist, err := m.UnitDist()
	if err != nil {
		return err
	}
	// retrieve Neighbourhood function
	nFn := tc.NeighbFn
	// perform iters number of learning iterations
	for i := 0; i < iters; i++ {
		// pick a random sample from dataset
		sample := data.RawRowView(r.Intn(rows))
		// no need to check for error here:
		// sample and codebook are not nil and have the same dimension
		bmu, _ := ClosestVec("euclidean", sample, m.codebook)
		// no need to check for errors:
		// LRate and Radius are checked by config validation
		lRate, _ := LRate(i, iters, tc.LDecay, tc.LRate)
		radius, _ := Radius(i, iters, tc.RDecay, tc.Radius)
		// pick the bmu unit distance row
		bmuDists := unitDist.RawRowView(bmu)
		// find units which are within the radius
		for i := 0; i < len(bmuDists); i++ {
			// bmu distance to i-th map unit
			dist := bmuDists[i]
			// we are within BMU radius
			if dist < radius {
				// update particular codebook vector
				m.seqUpdateCbVec(i, sample, lRate, radius, dist, nFn)
			}
		}
	}

	return nil
}

// seqUpdateCbVec updates codebook vector on row cbIdx given the learning rate l,
// radius r, distance d and neihgbourhood function nFn
func (m *Map) seqUpdateCbVec(cbIdx int, vec []float64, l, r, d float64, nFn NeighbFunc) {
	// pick codebook vector that should be updated
	cbVec := m.codebook.RawRowView(cbIdx)
	mul := l
	// Update codebook vector element by element
	for i := 0; i < len(cbVec); i++ {
		// nFn returns 1 for d == 0; skipping this case will save us some CPU time
		if d > 0.0 {
			mul *= nFn(d, r)
		}
		cbVec[i] = cbVec[i] + mul*(vec[i]-cbVec[i])
	}
}

// batchConfig holds batch training configuration
type batchConfig struct {
	// tc is SOM training configuration
	tc *TrainConfig
	// iters is a number of batch iterations
	iters int
}

// batchResult holds results of batch algorithm for a particular data input batch
type batchResult struct {
	// vecs is a slice of nghb scaled data vectors
	vecs [][]float64
	// nghbs is a slice of BMU neighbourhoods
	nghbs []float64
}

// batchTrain runs batch SOM training on a given data set
func (m *Map) batchTrain(tc *TrainConfig, data *mat.Dense, iters int) error {
	cbRows, _ := m.codebook.Dims()
	rows, _ := data.Dims()
	// batchConfig holds training config and number of iterations
	bc := &batchConfig{
		tc:    tc,
		iters: iters,
	}
	// calculate unit distances
	unitDist, err := m.UnitDist()
	if err != nil {
		return err
	}
	// number of worker goroutines
	workers := runtime.NumCPU()
	// evenly distribute batch work between workers
	workerBatch := rows / workers
	// train for a number of iterations
	for i := 0; i < iters; i++ {
		// reset from index and input count
		from := 0
		count := workerBatch
		// create batch results channel
		results := make(chan *batchResult, workers*40)
		wg := &sync.WaitGroup{}
		// start worker goroutines
		for j := 0; j < workers; j++ {
			// from is data matrix row pointer
			from = j * workerBatch
			// last worker will work through the batch reminder
			if j == workers-1 {
				count += rows % workers
			}
			// if we go over the number of rows adjust bSamples
			if from+count > rows {
				count = rows - from
			}
			wg.Add(1)
			go m.processBatch(results, wg, bc, unitDist, data, from, count, i)
		}
		// wait for workers to finish and close the result channel
		go func() {
			wg.Wait()
			close(results)
		}()
		// collect batch results from all workers
		vecs := make([][]float64, cbRows)
		nghbs := make([]float64, cbRows)
		for result := range results {
			for k := 0; k < len(result.vecs); k++ {
				if result.vecs[k] != nil {
					if vecs[k] != nil {
						for l := 0; l < len(vecs[k]); l++ {
							vecs[k][l] += result.vecs[k][l]
						}
					} else {
						vecs[k] = result.vecs[k]
					}
					nghbs[k] += result.nghbs[k]
				}
			}
		}
		// update codebook vectors
		for k := 0; k < cbRows; k++ {
			if vecs[k] != nil {
				for l := 0; l < len(vecs[k]); l++ {
					vecs[k][l] = vecs[k][l] / nghbs[k]
				}
				m.codebook.SetRow(k, vecs[k])
			}
		}
	}

	return nil
}

// processRow processes data rows and sends tehm down the results channel
func (m Map) processBatch(res chan<- *batchResult, wg *sync.WaitGroup,
	bc *batchConfig, unitDist, data *mat.Dense, from, count, iter int) {
	// allocate codebook vectors and neighbourhoods
	rows, _ := m.codebook.Dims()
	vecs := make([][]float64, rows)
	nghbs := make([]float64, rows)
	// retrieve Neighbourhood function
	nFn := bc.tc.NeighbFn
	// iterate through the whole batch
	for i := from; i < count+from; i++ {
		row := data.RawRowView(i)
		// find codebook BMU for this data row
		bmu, _ := ClosestVec("euclidean", row, m.codebook)
		// calculate radius for this iteration
		radius, _ := Radius(iter, bc.iters, bc.tc.RDecay, bc.tc.Radius)
		// pick the BMU's distance row
		bmuDists := unitDist.RawRowView(bmu)
		for j := 0; j < len(bmuDists); j++ {
			// bmu distance to i-th map unit
			dist := bmuDists[j]
			// when in BMU radius, scale and add to all neighbourhood vecs
			if dist < radius {
				// calculate neighbourhood function
				nghb := nFn(dist, radius)
				if vecs[j] != nil {
					for k := 0; k < len(vecs[j]); k++ {
						vecs[j][k] += nghb * row[k]
					}
				} else {
					vecs[j] = make([]float64, len(row))
					for k := 0; k < len(vecs[j]); k++ {
						vecs[j][k] = nghb * row[k]
					}
				}
				nghbs[j] += nghb
			}
		}
	}
	// send batchResult down results channel
	res <- &batchResult{vecs: vecs, nghbs: nghbs}
	wg.Done()
}
