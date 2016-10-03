package som

import "fmt"

func RunDeltaTests() {
	configDeltaTests()
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
