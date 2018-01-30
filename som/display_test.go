package som

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestUMatrixSVG(t *testing.T) {
	assert := assert.New(t)

	const svg = `<h1>Done</h1><svg width="120" height="120"><polygon points="35.000000,24.433757 10.000000,38.867513 -15.000000,24.433757 -15.000000,-4.433757 10.000000,-18.867513 35.000000,-4.433757 35.000000,24.433757 " style="fill:rgb(255,255,255);stroke:black;stroke-width:1"></polygon><polygon points="60.000000,67.735027 35.000000,82.168784 10.000000,67.735027 10.000000,38.867513 35.000000,24.433757 60.000000,38.867513 60.000000,67.735027 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><polygon points="85.000000,24.433757 60.000000,38.867513 35.000000,24.433757 35.000000,-4.433757 60.000000,-18.867513 85.000000,-4.433757 85.000000,24.433757 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><polygon points="110.000000,67.735027 85.000000,82.168784 60.000000,67.735027 60.000000,38.867513 85.000000,24.433757 110.000000,38.867513 110.000000,67.735027 " style="fill:rgb(255,255,255);stroke:black;stroke-width:1"></polygon></svg>`

	mUnits := mat.NewDense(4, 2, []float64{
		0.0, 0.0,
		0.0, 0.1,
		1.0, 1.0,
		1.0, 1.1,
	})
	coordDims := []int{2, 2}
	uShape := "hexagon"
	title := "Done"
	writer := bytes.NewBufferString("")

	UMatrixSVG(mUnits, coordDims, uShape, title, writer, make(map[int]int))

	assert.Equal(svg, writer.String())
	// make sure there is at least one fully black element
	assert.True(strings.Contains(svg, "rgb(0,0,0)"))
	// make sure there is at least one fully white element
	assert.True(strings.Contains(svg, "rgb(255,255,255)"))
}

func TestUMatrixSVGWithClusters(t *testing.T) {
	assert := assert.New(t)

	const svg = `<h1>Done</h1><svg width="70" height="120"><polygon points="35.000000,35.000000 35.000000,-15.000000 -15.000000,-15.000000 -15.000000,35.000000 35.000000,35.000000 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><text x="-2.5" y="22.5">0</text><polygon points="35.000000,85.000000 35.000000,35.000000 -15.000000,35.000000 -15.000000,85.000000 35.000000,85.000000 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><text x="-2.5" y="72.5">1</text></svg>`

	mUnits := mat.NewDense(2, 2, []float64{
		0.0, 0.0,
		1.0, 1.0,
	})
	coordDims := []int{2, 1}
	uShape := "rectangle"
	title := "Done"
	writer := bytes.NewBufferString("")

	classes := map[int]int{
		0: 0,
		1: 1,
	}
	UMatrixSVG(mUnits, coordDims, uShape, title, writer, classes)

	assert.Equal(svg, writer.String())
	// make sure there is at least one text element
	assert.True(strings.Contains(svg, "<text "))
}
