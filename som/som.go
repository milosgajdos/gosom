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

type CodebookInitFunc func(*mat64.Dense, []int) (*mat64.Dense, error)

type CoordsInitFunc func(string, []int) (*mat64.Dense, error)

type NeighbFunc func(float64, float64) float64

type Map struct {
	codebook *mat64.Dense
	unitDist *mat64.Dense
	bmus     map[int]int
}

func NewMap(c *MapConfig, data *mat64.Dense) (*Map, error) {
	if data == nil {
		return nil, fmt.Errorf("Invalid input data: %v\n", data)
	}
	if err := validateMapConfig(c); err != nil {
		return nil, err
	}
	mUnits := utils.IntProduct(c.Dims)
	if mUnits <= 1 {
		return nil, fmt.Errorf("Incorrect map size dimensions: %v\n", c.Dims)
	}
	codebook, err := c.InitFunc(data, c.Dims)
	if err != nil {
		return nil, err
	}
	gridCoords, err := GridCoords(c.UShape, c.Dims)
	if err != nil {
		return nil, err
	}
	unitDist, err := DistanceMx("euclidean", gridCoords)
	if err != nil {
		return nil, err
	}
	bmus := make(map[int]int)
	return &Map{
		codebook: codebook,
		unitDist: unitDist,
		bmus:     bmus,
	}, nil
}

func (m Map) Codebook() *mat64.Dense {
	return m.codebook
}

func (m Map) UnitDist() *mat64.Dense {
	return m.unitDist
}

func (m Map) BMUs() map[int]int {
	return m.bmus
}

func (m *Map) MarshalTo(format string, w io.Writer) (int, error) {
	switch format {
	case "gonum":
		return m.codebook.MarshalBinaryTo(w)
	}
	return 0, fmt.Errorf("Unsupported format: %s\n", format)
}

func (m Map) UMatrixOut(format, title string, w io.Writer) error {
	return nil
}

func (m *Map) Train(c *TrainConfig, data *mat64.Dense, iters int) error {
	if iters <= 0 {
		return fmt.Errorf("Invalid number of iterations: %d\n", iters)
	}
	if data == nil {
		return fmt.Errorf("Invalid data supplied: %v\n", data)
	}
	if err := validateTrainConfig(c); err != nil {
		return err
	}
	switch c.Method {
	case "seq":
		return m.seqTrain(c, data, iters)
	case "batch":
		return m.batchTrain(c, data, iters)
	}

	return nil
}

func (m *Map) seqTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	rows, _ := data.Dims()
	rSrc := rand.NewSource(time.Now().UnixNano())
	r := rand.New(rSrc)
	neighbFn := Neighb[tc.NeighbFn]
	for i := 0; i < iters; i++ {
		sample := data.RowView(r.Intn(rows))
		bmu, _ := ClosestVec("euclidean", sample, m.codebook)
		lRate, _ := LRate(i, iters, tc.LDecay, tc.LRate)
		radius, _ := Radius(i, iters, tc.RDecay, tc.Radius)
		bmuDists := m.unitDist.RowView(bmu)
		for i := 0; i < bmuDists.Len(); i++ {
			dist := bmuDists.At(i, 0)
			if dist < radius {
				m.seqUpdateCbVec(i, sample, lRate, radius, dist, neighbFn)
			}
		}
	}

	return nil
}

func (m *Map) seqUpdateCbVec(cbIdx int, vec *mat64.Vector, l, r, d float64, nFn NeighbFunc) {
	cbVec := m.codebook.RowView(cbIdx)
	diff := mat64.NewVector(cbVec.Len(), nil)
	diff.AddScaledVec(vec, -1.0, cbVec)
	mul := l
	if d > 0.0 {
		mul *= nFn(d, r)
	}
	cbVec.AddScaledVec(cbVec, mul, diff)
}

type batchConfig struct {
	tc    *TrainConfig
	iters int
}

type batchResult struct {
	vec  *mat64.Vector
	nghb float64
	idx  int
}

func (m *Map) batchTrain(tc *TrainConfig, data *mat64.Dense, iters int) error {
	cbRows, _ := m.codebook.Dims()
	rows, _ := data.Dims()
	bSize := cbRows
	if rows < cbRows {
		bSize = rows
	}
	bIters := rows / bSize
	if rows%bSize != 0 {
		bIters++
	}
	fmt.Println("batch iters", bIters)
	bc := &batchConfig{
		tc:    tc,
		iters: bIters,
	}
	workers := runtime.NumCPU()
	for i := 0; i < iters; i++ {
		count := bSize
		iter := 0
		for j := 0; j < rows; j += bSize {
			results := make(chan *batchResult, workers*4)
			wg := &sync.WaitGroup{}
			// Changes here: bSize/workers etc.
			for k := 0; k < workers; k++ {
				from := j + k*(bSize/workers)
				if from+bSize > rows {
					count = rows - from
				}
				wg.Add(1)
				go m.processBatch(results, wg, bc, data, from, count, iter)
				if count < bSize {
					break
				}
			}
			iter++
			go func() {
				wg.Wait()
				close(results)
			}()
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
			for k := 0; k < cbRows; k++ {
				if cbVecs[k] != nil {
					cbVecs[k].ScaleVec(1.0/nghbs[k], cbVecs[k])
					m.codebook.SetRow(k, cbVecs[k].RawVector().Data)
				}
			}
		}
	}

	return nil
}

func (m Map) processBatch(res chan<- *batchResult, wg *sync.WaitGroup,
	bc *batchConfig, data *mat64.Dense, from, count, iter int) {
	fmt.Println("ITER", iter)
	_, cols := data.Dims()
	neighbFn := Neighb[bc.tc.NeighbFn]
	for i := from; i < count+from; i++ {
		row := data.RowView(i)
		bmu, _ := ClosestVec("euclidean", row, m.codebook)
		radius, _ := Radius(iter, bc.iters, bc.tc.RDecay, bc.tc.Radius)
		bmuDists := m.unitDist.RowView(bmu)
		for j := 0; j < bmuDists.Len(); j++ {
			dist := bmuDists.At(j, 0)
			if dist < radius {
				nghb := neighbFn(dist, radius)
				vec := mat64.NewVector(cols, nil)
				vec.CopyVec(row)
				vec.ScaleVec(nghb, vec)
				res <- &batchResult{vec: vec, nghb: nghb, idx: j}
			}
		}
	}
	wg.Done()
}
