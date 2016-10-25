package som

import (
	"encoding/xml"
	"fmt"
	"io"
	"math"

	"github.com/gonum/matrix/mat64"
)

type rowWithDist struct {
	Row  int
	Dist float64
}

type h1 struct {
	XMLName xml.Name `xml:"h1"`
	Title   string   `xml:",innerxml"`
}

type polygon struct {
	XMLName xml.Name `xml:"polygon"`
	Points  []byte   `xml:"points,attr"`
	Style   string   `xml:"style,attr"`
}

type svgElement struct {
	XMLName  xml.Name `xml:"svg"`
	Width    float64  `xml:"width,attr"`
	Height   float64  `xml:"height,attr"`
	Polygons []interface{}
}

type textElement struct {
	XMLName xml.Name `xml:"text"`
	X       float64  `xml:"x,attr"`
	Y       float64  `xml:"y,attr"`
	Text    string   `xml:",innerxml"`
}

//var colors = [][]int{{255, 255, 255}}

var colors = [][]int{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255}}

// Creates an SVG representation of the U-Matrix of the given codebook.
// codebook is the codebook we're displaying the U-Matrix for,
// coordsDims are the dimensions of the grid,
// uShape is the shape of the grid,
// title is the title of the output SVG,
// and writer is the io.Writter to write the output SVG to.
func UMatrixSVG(codebook *mat64.Dense, coordsDims []int, uShape string, title string, writer io.Writer, clusters map[int]int) error {
	xmlEncoder := xml.NewEncoder(writer)
	// array to hold the xml elements
	elems := []interface{}{h1{Title: title}}

	rows, _ := codebook.Dims()
	distMat, err := DistanceMx("euclidean", codebook)
	if err != nil {
		return err
	}
	coords, err := GridCoords(uShape, coordsDims)
	if err != nil {
		return err
	}
	coordsDistMat, err := DistanceMx("euclidean", coords)
	if err != nil {
		return err
	}

	// get maximum distance between codebook vectors
	// maximum distance will be rgb(0,0,0)
	maxDistance := mat64.Max(distMat)

	// function to scale the coord grid to something visible
	const MUL = 50.0
	const OFF = 10.0
	scale := func(x float64) float64 { return MUL*x + OFF }

	svgElem := svgElement{
		Width:    float64(coordsDims[1])*MUL + 2*OFF,
		Height:   float64(coordsDims[0])*MUL + 2*OFF,
		Polygons: make([]interface{}, rows*2),
	}
	elems = append(elems, svgElem)
	for row := 0; row < rows; row++ {
		coord := coords.RowView(row)
		cbVec := codebook.RowView(row)
		avgDistance := 0.0
		// this is a rough approximation of the notion of neighbor grid coords
		allRowsInRadius := allRowsInRadius(row, math.Sqrt2*1.01, coordsDistMat)
		for _, rwd := range allRowsInRadius {
			if rwd.Dist > 0.0 {
				otherMu := codebook.RowView(rwd.Row)
				cbvDist, err := Distance("euclidean", cbVec, otherMu)
				if err != nil {
					return err
				}

				avgDistance += cbvDist
			}
		}
		avgDistance /= float64(len(allRowsInRadius) - 1)
		var colorMask []int
		if clusters[row] == -1 {
			colorMask = []int{255, 255, 255}
		} else {
			colorMask = colors[clusters[row]%len(colors)]
		}
		colorMul := (1.0 - avgDistance/maxDistance)
		r := int(colorMul * float64(colorMask[0]))
		g := int(colorMul * float64(colorMask[1]))
		b := int(colorMul * float64(colorMask[2]))
		polygonCoords := ""
		x := scale(coord.At(0, 0))
		y := scale(coord.At(1, 0))
		xOffset := 0.5 * MUL
		yOffset := 0.5 * MUL
		// hexagon has a different yOffset
		if uShape == "hexagon" {
			yOffset = math.Sqrt(0.75) / 2.0 * MUL
		}
		// draw a box around the current coord
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y-yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y-yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y+yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)

		svgElem.Polygons[row*2] = polygon{
			Points: []byte(polygonCoords),
			Style:  fmt.Sprintf("fill:rgb(%d,%d,%d);stroke:black;stroke-width:1", r, g, b),
		}

		svgElem.Polygons[row*2+1] = textElement{
			X:    x - 0.25*MUL,
			Y:    y + 0.25*MUL,
			Text: fmt.Sprintf("%d", clusters[row]),
		}
	}

	xmlEncoder.Encode(elems)
	xmlEncoder.Flush()

	return nil
}

func allRowsInRadius(selectedRow int, radius float64, distMatrix *mat64.Dense) []rowWithDist {
	rowsInRadius := []rowWithDist{}
	for i, dist := range distMatrix.RowView(selectedRow).RawVector().Data {
		if dist < radius {
			rowsInRadius = append(rowsInRadius, rowWithDist{Row: i, Dist: dist})
		}
	}
	return rowsInRadius
}
