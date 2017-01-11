package main

import (
	"fmt"
	"log"
	"os"

	"github.com/milosgajdos83/gosom/som"
)

func main() {
	// SOM configuration
	grid := &som.GridConfig{
		Dims:   mdims,
		Type:   grid,
		UShape: ushape,
	}
	cb := &som.CbConfig{
		Dim:      dim,
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
		Method:   training,
		Radius:   radius,
		RDecay:   rdecay,
		NeighbFn: neighb,
		LRate:    lrate,
		LDecay:   ldecay,
	}
	if err := m.Train(trainCfg, data, iters); err != nil {
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
