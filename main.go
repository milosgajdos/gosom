package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/milosgajdos83/gosom/pkg/dataset"
	"github.com/milosgajdos83/gosom/pkg/utils"
	"github.com/milosgajdos83/gosom/som"
)

var (
	// path to input data set
	input string
	// classification information flag
	clsinfo bool
	// feature scaling flag
	scale bool
	// map dimensions: 2D only [for now]
	dims string
	// map grid type: planar
	grid string
	// training method: seq, batch
	training string
	// map unit shape type: hexagon, rectangle
	ushape string
	// initial SOM unit neihbourhood radius
	radius float64
	// radius decay strategy: lin, exp
	rdecay string
	// neighbourhood func: gaussian, bubble, mexican
	neighb string
	// initial SOM learning rate
	lrate float64
	// learning rate decay strategy: lin, exp
	ldecay string
	// path to umatrix visualization
	umxout string
	// path to saved SOM model
	output string
	// number of training iterations
	iters int
)

func init() {
	flag.StringVar(&input, "input", "", "Path to input data set")
	flag.BoolVar(&clsinfo, "clsinfo", false, "Dataset has classification information in a .cls file of same name as input")
	flag.BoolVar(&scale, "scale", false, "Request data scaling")
	flag.StringVar(&dims, "dims", "", "comma-separated SOM dimensions")
	flag.StringVar(&grid, "grid", "planar", "SOM grid")
	flag.StringVar(&ushape, "ushape", "hexagon", "SOM map unit shape")
	flag.StringVar(&training, "training", "seq", "SOM training method")
	flag.Float64Var(&radius, "radius", 0.0, "SOM neihbourhood starting radius")
	flag.StringVar(&rdecay, "rdecay", "lin", "Radius decay strategy")
	flag.StringVar(&neighb, "neighb", "gaussian", "SOM neighbourhood function")
	flag.Float64Var(&lrate, "lrate", 0.0, "SOM initial learning rate")
	flag.StringVar(&ldecay, "ldecay", "lin", "Learning rate decay strategy")
	flag.StringVar(&umxout, "umxout", "", "Path to u-matrix output visualization")
	flag.StringVar(&output, "output", "", "Path to serialize the learnt SOM model")
	flag.IntVar(&iters, "iters", 1000, "Number of training iterations")
}

func parseCliFlags() error {
	// parse cli flags
	flag.Parse()
	// path to input data is mandatory
	if input == "" {
		return fmt.Errorf("Invalid path to input data: %s\n", input)
	}
	// number of iterations mus tbe positive integer
	if iters <= 0 {
		return fmt.Errorf("Invalid number of training iterations: %d", iters)
	}
	return nil
}

func main() {
	// parse cli flags; exit with non-zero return code on error
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
		os.Exit(1)
	}
	// parse SOM grid dimensions
	mdims, err := utils.ParseDims(dims)
	if err != nil {
		fmt.Printf("Error parsing grid dimensions: %s\n", err)
		os.Exit(1)
	}
	// load data set from a file in provided path
	ds, err := dataset.New(input, clsinfo)
	if err != nil {
		fmt.Printf("Unable to load Data Set: %s\n", err)
		os.Exit(1)
	}
	// scale input data if requested
	data := ds.Data()
	if scale {
		data = ds.Scale()
	}
	// SOM configuration
	mConfig := &som.MapConfig{
		Dims:     mdims,
		InitFunc: som.RandInit,
		Grid:     grid,
		UShape:   ushape,
	}
	// create new SOM map
	smap, err := som.NewMap(mConfig, data)
	if err != nil {
		fmt.Printf("Failed to create new SOM: %s\n", err)
		os.Exit(1)
	}
	// training configuration
	tConfig := &som.TrainConfig{
		Method:   training,
		Radius:   radius,
		RDecay:   rdecay,
		NeighbFn: neighb,
		LRate:    lrate,
		LDecay:   ldecay,
	}
	// run SOM training
	if err := smap.Train(tConfig, data, iters); err != nil {
		fmt.Printf("Training failed: %s\n", err)
		os.Exit(1)
	}
	// if output is not empty save map to a file
	if output != "" {
		file, err := os.Open(output)
		if err != nil {
			fmt.Printf("Output file error: %s\n", err)
			os.Exit(1)
		}
		defer file.Close()
		// save the model
		if _, err := smap.MarshalTo("gonum", file); err != nil {
			fmt.Printf("Failed to save model: %s\n", err)
			os.Exit(1)
		}
	}

	// if umxout provided create U-matrix
	if umxout != "" {
		file, err := os.Create(umxout)
		if err != nil {
			fmt.Printf("Umatrix file error: %s\n", err)
			os.Exit(1)
		}
		defer file.Close()
		smap.UMatrixOut("svg", input, mConfig, file, ds)
	}
}
