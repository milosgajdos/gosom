package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/milosgajdos83/gosom/pkg/dataset"
	"github.com/milosgajdos83/gosom/pkg/utils"
	"github.com/milosgajdos83/gosom/som"
)

const (
	cliname = "gosom"
)

var (
	// path to input data set
	input string
	// path to classification file for the data set
	cls string
	// feature scaling flag
	scale bool
	// coma separated map dimensions: 2D only [for now]
	dims string
	// map grid type: planar
	grid string
	// map unit shape: hexagon, rectangle
	ushape string
	// initial unit neihbourhood radius
	radius float64
	// radius decay strategy: lin, exp
	rdecay string
	// neighbourhood func: gaussian, bubble, mexican
	neighb string
	// initial learning rate
	lrate float64
	// learning rate decay strategy: lin, exp
	ldecay string
	// path to umatrix visualization
	umatrix string
	// path to saved model
	output string
	// training method: seq, batch
	training string
	// number of training iterations
	iters int
)

func init() {
	flag.StringVar(&input, "input", "", "Path to input data set")
	flag.StringVar(&cls, "cls", "", "Path to input data set classification file")
	flag.BoolVar(&scale, "scale", false, "Request data scaling")
	flag.StringVar(&dims, "dims", "", "comma-separated SOM grid dimensions")
	flag.StringVar(&grid, "grid", "planar", "Type of SOM grid")
	flag.StringVar(&ushape, "ushape", "hexagon", "SOM map unit shape")
	flag.StringVar(&neighb, "neighb", "gaussian", "SOM neighbourhood function")
	flag.Float64Var(&radius, "radius", 0.0, "SOM neighbourhood initial radius")
	flag.StringVar(&rdecay, "rdecay", "lin", "Radius decay strategy")
	flag.Float64Var(&lrate, "lrate", 0.0, "SOM initial learning rate")
	flag.StringVar(&ldecay, "ldecay", "lin", "Learning rate decay strategy")
	flag.StringVar(&umatrix, "umatrix", "", "Path to u-matrix output visualization")
	flag.StringVar(&output, "output", "", "Path to store trained SOM model")
	flag.StringVar(&training, "training", "seq", "SOM training method")
	flag.IntVar(&iters, "iters", 1000, "Number of training iterations")
	// disable timestamps and set prefix
	log.SetFlags(0)
	log.SetPrefix("[ " + cliname + " ] ")
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

func saveModel(m *som.Map, format, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	// save the model
	if _, err := m.MarshalTo(format, file); err != nil {
		return err
	}

	return nil
}

func saveUMatrix(m *som.Map, format, title, path string, c *som.MapConfig, d *dataset.DataSet) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return m.UMatrix(format, title, c, d, file)
}

func main() {
	// parse cli flags
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	// parse SOM grid dimensions
	mdims, err := utils.ParseDims(dims)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	log.Printf("Loading data set %s", input)
	// load input data set from a file in provided path
	ds, err := dataset.New(input, cls)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// scale features in input data if requested
	data := ds.Data()
	if scale {
		log.Printf("Attempting feature scaling")
		data = ds.Scale()
	}
	_, dim := data.Dims()
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
	// create new SOM
	log.Printf("Creating new SOM. Dimensions: %v, Grid Type: %s, Unit shape: %s",
		mapCfg.Grid.Dims, mapCfg.Grid.Type, mapCfg.Grid.UShape)
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
	// run SOM training
	log.Printf("Starting SOM training. Method: %s, iterations: %d", trainCfg.Method, iters)
	t0 := time.Now()
	if err := m.Train(trainCfg, data, iters); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	d := time.Since(t0)
	//log.Printf("Training completed. Duration: [%.3f]ms", float64(d)/float64(time.Millisecond))
	log.Printf("Training successfully completed. Duration: %v", d)
	// if output is not empty save map model to a file
	if output != "" {
		log.Printf("Saving trained model to %s", output)
		if err := saveModel(m, "gonum", output); err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
	}
	// if umatrix provided create U-matrix
	if umatrix != "" {
		log.Printf("Saving U-Matrix to %s", umatrix)
		if err := saveUMatrix(m, "svg", "U-Matrix", umatrix, mapCfg, ds); err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
	}

	// ======= SOM QUALITY measures =======
	// Quantization error
	qe, err := m.QuantError(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	log.Printf("Quantization Error: %f\n", qe)
	// Topographic product
	tp, err := m.TopoProduct()
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	log.Printf("Topographic Product: %f\n", tp)
	// Topographice error
	te, err := m.TopoError(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	log.Printf("Topographic Error: %f\n", te)
}
