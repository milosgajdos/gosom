package som

import (
	"bytes"
	"strings"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestUMatrixSVG(t *testing.T) {
	assert := assert.New(t)

	const svg = `<h1>Done</h1><svg width="120" height="120"><polygon points="35.000000,31.650635 35.000000,-11.650635 -15.000000,-11.650635 -15.000000,31.650635 35.000000,31.650635 " style="fill:rgb(255,255,255);stroke:black;stroke-width:1"></polygon><polygon points="60.000000,74.951905 60.000000,31.650635 10.000000,31.650635 10.000000,74.951905 60.000000,74.951905 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><polygon points="85.000000,31.650635 85.000000,-11.650635 35.000000,-11.650635 35.000000,31.650635 85.000000,31.650635 " style="fill:rgb(0,0,0);stroke:black;stroke-width:1"></polygon><polygon points="110.000000,74.951905 110.000000,31.650635 60.000000,31.650635 60.000000,74.951905 110.000000,74.951905 " style="fill:rgb(255,255,255);stroke:black;stroke-width:1"></polygon></svg>`

	mUnits := mat64.NewDense(4, 2, []float64{
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

	mUnits := mat64.NewDense(2, 2, []float64{
		0.0, 0.0,
		1.0, 1.0,
	})
	coordDims := []int{2, 1}
	uShape := "rectangle"
	title := "Done"
	writer := bytes.NewBufferString("")

	clusters := map[int]int{
		0: 0,
		1: 1,
	}
	UMatrixSVG(mUnits, coordDims, uShape, title, writer, clusters)

	assert.Equal(svg, writer.String())
	// make sure there is at least one text element
	assert.True(strings.Contains(svg, "<text "))
}
