package som

import (
	"bytes"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestUMatrixSVG(t *testing.T) {
	assert := assert.New(t)

	const svg = `<h1>Done</h1><svg width="60" height="60"><polygon points="20.000000,18.660254 20.000000,1.339746 0.000000,1.339746 0.000000,18.660254 20.000000,18.660254 " style="fill:rgb(125,125,125);stroke:black;stroke-width:1"></polygon><polygon points="30.000000,35.980762 30.000000,18.660254 10.000000,18.660254 10.000000,35.980762 30.000000,35.980762 " style="fill:rgb(91,91,91);stroke:black;stroke-width:1"></polygon><polygon points="40.000000,18.660254 40.000000,1.339746 20.000000,1.339746 20.000000,18.660254 40.000000,18.660254 " style="fill:rgb(91,91,91);stroke:black;stroke-width:1"></polygon><polygon points="50.000000,35.980762 50.000000,18.660254 30.000000,18.660254 30.000000,35.980762 50.000000,35.980762 " style="fill:rgb(125,125,125);stroke:black;stroke-width:1"></polygon></svg>`

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

	UMatrixSVG(mUnits, coordDims, uShape, title, writer)

	assert.Equal(svg, writer.String())
}
