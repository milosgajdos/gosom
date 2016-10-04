package som

import "fmt"
import "github.com/gonum/matrix/mat64"

func RunDeltaTests() {
	printSection("Config")
	configDeltaTests()
	printSection("Grid")
	gridDeltaTests()
	printSection("SOM")
	somDeltaTests()
}

func somDeltaTests() {
	printSection("New Map")
	// default config should not throw any errors
	printD(NewMap(defaultConfig(), inputData()))
	// incorrect config
	origLcool := defaultConfig().LCool
	defaultConfig().LCool = "foobar"
	printD(NewMap(defaultConfig(), inputData()))
	defaultConfig().LCool = origLcool
	// incorrect init matrix
	printD(NewMap(defaultConfig(), nil))
	// incorrect number of map units
	origDims := defaultConfig().Dims
	defaultConfig().Dims = []int{0, 0}
	printD(NewMap(defaultConfig(), inputData()))
	defaultConfig().Dims = origDims

	printSection("Codebook")
	// default config should not throw any errors
	m, err := NewMap(defaultConfig(), inputData())
	printD(m, err)
	printD(m.Codebook())
	printD(m.Codebook().Dims())

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
	printSection("RandInit")

	min1, max1 := 1.2, 4.5
	min2, max2 := 3.4, 6.7
	inMx := mat64.NewDense(2, 2, []float64{min1, min2, max1, max2})
	rows := []int{2, 2}
	printD(inMx)

	// initialize random matrix
	printD(RandInit(inMx, rows))

	// nil input matrix
	printD(RandInit(nil, rows))
	// negative number of rows
	printD(RandInit(inMx, []int{-2, 2}))
	// empty matrix
	emptyMx := mat64.NewDense(0, 0, nil)
	printD(RandInit(emptyMx, []int{2, 3}))

	printSection("GridDims")

	uShape := "hexagon"
	// 1D data with more than one sample
	data := mat64.NewDense(2, 1, []float64{2, 3})
	printD(GridDims(data, uShape))
	// 2D data with one sample
	data = mat64.NewDense(1, 2, []float64{2, 3})
	printD(GridDims(data, uShape))
	// 2D+ data with more than one sample
	data = mat64.NewDense(6, 4, []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2,
		5.4, 3.9, 1.7, 0.4,
	})
	printD(GridDims(data, uShape))
	// data matrix can't be nil
	printD(GridDims(nil, uShape))

	printSection("LinInit")
	inMx = mat64.NewDense(6, 4, []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2,
		5.4, 3.9, 1.7, 0.4,
	})

	xDim, yDim := 5, 2
	linMx, err := LinInit(inMx, []int{xDim, yDim})
	printD(linMx, err)
	// check if the dimensions are correct munits x datadim
	printD(linMx.Dims())
	// data is nil
	printD(LinInit(nil, []int{1, 2}))
	// nil dimensions
	printD(LinInit(inMx, nil))
	// non positive dimensions supplied
	printD(LinInit(inMx, []int{-1, 2}))
	// insufficient number of samples
	inMx = mat64.NewDense(1, 2, []float64{1, 1})
	printD(LinInit(inMx, []int{5, 2}))

	printSection("GridCoords")

	// hexagon shape
	dims := []int{4, 2}
	printD(GridCoords("hexagon", dims))
	// rectangle shape
	dims = []int{3, 2}
	printD(GridCoords("rectangle", dims))
	// incorrect units shape
	printD(GridCoords("fooshape", []int{2, 2}))
	// nil dimensions
	printD(GridCoords("hexagon", nil))
	// unsupported number of dimensions
	printD(GridCoords("hexagon", []int{1, 2, 3, 4}))
	// negative plane dimensions
	printD(GridCoords("hexagon", []int{-1, 2}))

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
	printSection("Default config")
	printD(validateConfig(defaultConfig()))

	printSection("Dims")
	for _, t := range [][]int{
		[]int{1},
		[]int{},
		[]int{1, 2},
		[]int{-1, 2},
	} {
		c := defaultConfig()
		c.Dims = t
		printD(validateConfig(c))
	}

	printSection("Grid")
	for _, t := range []string{
		"planar",
		"foobar",
	} {
		c := defaultConfig()
		c.Grid = t
		printD(validateConfig(c))
	}

	printSection("UShape")
	for _, t := range []string{
		"hexagon",
		"foobar",
		"rectangle",
	} {
		c := defaultConfig()
		c.UShape = t
		printD(validateConfig(c))
	}

	printSection("Radius")
	for _, t := range []int{
		1,
		-10,
		0,
	} {
		c := defaultConfig()
		c.Radius = t
		printD(validateConfig(c))
	}

	printSection("RCool")
	for _, t := range []string{
		"lin",
		"foobar",
		"exp",
		"inv",
	} {
		c := defaultConfig()
		c.RCool = t
		printD(validateConfig(c))
	}

	printSection("NeighbFn")
	for _, t := range []string{
		"gaussian",
		"foobar",
		"mexican",
		"bubble",
	} {
		c := defaultConfig()
		c.NeighbFn = t
		printD(validateConfig(c))
	}

	printSection("LRate")
	for _, t := range []int{
		1,
		-10,
		0,
	} {
		c := defaultConfig()
		c.LRate = t
		printD(validateConfig(c))
	}

	printSection("LCool")
	for _, t := range []string{
		"lin",
		"exp",
		"foobar",
		"inv",
	} {
		c := defaultConfig()
		c.LCool = t
		printD(validateConfig(c))
	}
}

func printSection(section string) {
	fmt.Printf("\n====== %s ======\n", section)
}

func printD(args ...interface{}) {
	fmt.Printf("%s\n", args...)
}
