package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"time"

	"image/color"
	"image/jpeg"
	"image/png"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gosom/pkg/utils"
	"github.com/milosgajdos83/gosom/som"
)

const (
	cliname = "gosom"
)

var (
	// path to input data set
	input string
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
	// path to saved model
	output string
	// training method: seq, batch
	training string
	// number of training iterations
	iters int
)

func init() {
	flag.StringVar(&input, "input", "", "Path to input data set")
	flag.StringVar(&dims, "dims", "", "comma-separated SOM grid dimensions")
	flag.StringVar(&grid, "grid", "planar", "Type of SOM grid")
	flag.StringVar(&ushape, "ushape", "hexagon", "SOM map unit shape")
	flag.StringVar(&neighb, "neighb", "gaussian", "SOM neighbourhood function")
	flag.Float64Var(&radius, "radius", 0.0, "SOM neighbourhood initial radius")
	flag.StringVar(&rdecay, "rdecay", "lin", "Radius decay strategy")
	flag.Float64Var(&lrate, "lrate", 0.0, "SOM initial learning rate")
	flag.StringVar(&ldecay, "ldecay", "lin", "Learning rate decay strategy")
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
		return fmt.Errorf("invalid path to input data: %s", input)
	}
	// output can't be empty
	if output == "" {
		return fmt.Errorf("invalid path to output data: %s", output)
	}
	// number of iterations mus tbe positive integer
	if iters <= 0 {
		return fmt.Errorf("invalid number of training iterations: %d", iters)
	}
	return nil
}

// ReadImage reads an image file in path and returns it as image.Image or fails with error
func ReadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// SaveImage saves img image in path or fails with error
func SaveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	switch filepath.Ext(path) {
	case ".jpeg":
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	case ".png":
		return png.Encode(f, img)
	}

	return fmt.Errorf("Unsupported image format: %s\n", filepath.Ext(path))
}

func Image2Data(img image.Image) *mat64.Dense {
	// get image bounds
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	// 4 dimensions: R, G, B, A
	data := mat64.NewDense(w*h, 4, nil)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			// convert 16 bit images to 8 bit color masks
			row := []float64{float64(r >> 8), float64(g >> 8), float64(b >> 8), float64(a >> 8)}
			data.SetRow(i, row)
			i++
		}
	}
	// scale data to 0-255 colors
	data.Scale(1/255.0, data)

	return data
}

func Data2Image(data *mat64.Dense, w, h int) image.Image {
	// create new RGB image
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	b := img.Bounds()
	// de-normalize the data
	data.Scale(255.0, data)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			row := data.RowView(i)
			clr := color.RGBA{uint8(row.At(0, 0)), uint8(row.At(1, 0)), uint8(row.At(2, 0)), uint8(row.At(3, 0))}
			img.Set(x, y, clr)
			i++
		}
	}

	return img
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
	// read test image
	img, err := ReadImage(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
	// convert image to data
	data := Image2Data(img)
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
	log.Printf("Training successfully completed. Duration: %v", d)
	// codebook vectors contains sorted colors
	imgData := m.Codebook().(*mat64.Dense)
	somImg := Data2Image(imgData, mdims[0], mdims[1])
	// save imaee
	if err := SaveImage(output, somImg); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
}
