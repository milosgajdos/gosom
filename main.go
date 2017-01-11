package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/som"
)

func main() {
	// make random data
	d := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	data := mat64.NewDense(5, 4, d)
	// SOM configuration
	grid := &som.GridConfig{
		Dims:   []int{2, 2},
		Type:   "planar",
		UShape: "hexagon",
	}
	cb := &som.CbConfig{
		Dim:      4,
		InitFunc: som.RandInit,
	}
	mapCfg := &som.MapConfig{
		Grid: grid,
		Cb:   cb,
	}
	// create new SOM§§
	m, err := som.NewMap(mapCfg, data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	// training configuration
	trainCfg := &som.TrainConfig{
		Method:   "seq",
		Radius:   500.0,
		RDecay:   "exp",
		NeighbFn: "gaussian",
		LRate:    0.5,
		LDecay:   "exp",
	}
	if err := m.Train(trainCfg, data, 300); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	// check quantization error
	qe, err := m.QuantError(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	log.Printf("Quantization Error: %f\n", qe)
}
