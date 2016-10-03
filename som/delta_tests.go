package som

import "fmt"

import "errors"

//import "os"

import "github.com/gonum/matrix/mat64"

//import "github.com/esemsch/gosom/pkg/utils"

func RunDeltaTests() {
	configDeltaTests()
	gridDeltaTests()
	somDeltaTests()
}

func somDeltaTests() {
	// default config should not throw any errors
	fmt.Println(NewMap(defaultConfig(), RandInit, inputData()))
	// incorrect config
	origLcool := defaultConfig().LCool
	defaultConfig().LCool = "foobar"
	fmt.Println(NewMap(defaultConfig(), RandInit, inputData()))
	defaultConfig().LCool = origLcool
	// when nil init function, use RandInit
	fmt.Println(NewMap(defaultConfig(), nil, inputData()))
	// incorrect init matrix
	fmt.Println(NewMap(defaultConfig(), RandInit, nil))
	// incorrect number of map units
	origDims := defaultConfig().Dims
	defaultConfig().Dims = []int{0, 0}
	fmt.Println(NewMap(defaultConfig(), RandInit, inputData()))
	defaultConfig().Dims = origDims
	// init func that always returns error
	fmt.Println(NewMap(defaultConfig(), func(inMx *mat64.Dense, rows int) (*mat64.Dense, error) { return nil, errors.New("Failed") }, inputData()))
}

func inputData() *mat64.Dense {
	data := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	return mat64.NewDense(5, 4, data)
}

func gridDeltaTests() {
	min1, max1 := 1.2, 4.5
	min2, max2 := 3.4, 6.7
	data := []float64{min1, min2, max1, max2}
	inMx := mat64.NewDense(2, 2, data)
	rows := 4
	fmt.Println(inMx)

	// initialize random matrix
	fmt.Println(RandInit(inMx, rows))

	// nil input matrix
	fmt.Println(RandInit(nil, rows))
	// negative number of rows
	fmt.Println(RandInit(inMx, -9))
	// empty matrix
	emptyMx := mat64.NewDense(0, 0, nil)
	fmt.Println(RandInit(emptyMx, 10))

}

func defaultConfig() *Config {
	return &Config{
		Dims:     []int{2, 3},
		Grid:     "planar",
		UShape:   "hexagon",
		Radius:   0,
		RCool:    "lin",
		NeighbFn: "gaussian",
		LRate:    0,
		LCool:    "lin",
	}
}

func configDeltaTests() {
	fmt.Println("Default config")
	fmt.Println(validateConfig(defaultConfig()))

	fmt.Println("Dims")
	for _, t := range [][]int{
		[]int{1},
		[]int{},
		[]int{1, 2},
		[]int{-1, 2},
	} {
		c := defaultConfig()
		c.Dims = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("Grid")
	for _, t := range []string{
		"planar",
		"foobar",
	} {
		c := defaultConfig()
		c.Grid = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("UShape")
	for _, t := range []string{
		"hexagon",
		"foobar",
		"rectangle",
	} {
		c := defaultConfig()
		c.UShape = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("Radius")
	for _, t := range []int{
		1,
		-10,
		0,
	} {
		c := defaultConfig()
		c.Radius = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("RCool")
	for _, t := range []string{
		"lin",
		"foobar",
		"exp",
		"inv",
	} {
		c := defaultConfig()
		c.RCool = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("NeighbFn")
	for _, t := range []string{
		"gaussian",
		"foobar",
		"mexican",
		"bubble",
	} {
		c := defaultConfig()
		c.NeighbFn = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("LRate")
	for _, t := range []int{
		1,
		-10,
		0,
	} {
		c := defaultConfig()
		c.LRate = t
		fmt.Println(validateConfig(c))
	}

	fmt.Println("LCool")
	for _, t := range []string{
		"lin",
		"exp",
		"foobar",
		"inv",
	} {
		c := defaultConfig()
		c.LCool = t
		fmt.Println(validateConfig(c))
	}
}
