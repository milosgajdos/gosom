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

var colors = [][]int{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255}}

// UMatrixSVG creates an SVG representation of the U-Matrix of the given codebook.
// It accepts the following parameters:
// codebook - the codebook we're displaying the U-Matrix for
// dims     - the dimensions of the map grid
// uShape   - the shape of the map grid
// title    - the title of the output SVG
// writer   - the io.Writter to write the output SVG to.
// classes  - if the classes are known (i.e. these are test data) they can be displayed providing the information in this map.
// The map is: codebook vector row -> class number. When classes are not known (i.e. running with real data), just provide an empty map
func UMatrixSVG(codebook *mat64.Dense, dims []int, uShape, title string, writer io.Writer, classes map[int]int) error {
	xmlEncoder := xml.NewEncoder(writer)
	// array to hold the xml elements
	elems := []interface{}{h1{Title: title}}

	rows, _ := codebook.Dims()
	distMat, err := DistanceMx("euclidean", codebook)
	if err != nil {
		return err
	}
	coords, err := GridCoords(uShape, dims)
	if err != nil {
		return err
	}
	coordsDistMat, err := DistanceMx("euclidean", coords)
	if err != nil {
		return err
	}

	umatrix := make([]float64, rows)
	maxDistance := -math.MaxFloat64
	minDistance := math.MaxFloat64
	for row := 0; row < rows; row++ {
		avgDistance := 0.0
		// this is a rough approximation of the notion of neighbor grid coords
		allRowsInRadius := allRowsInRadius(row, math.Sqrt2*1.01, coordsDistMat)
		for _, rwd := range allRowsInRadius {
			if rwd.Dist > 0.0 {
				avgDistance += distMat.At(row, rwd.Row)
			}
		}
		avgDistance /= float64(len(allRowsInRadius) - 1)
		umatrix[row] = avgDistance
		if avgDistance > maxDistance {
			maxDistance = avgDistance
		}
		if avgDistance < maxDistance {
			minDistance = avgDistance
		}
	}

	// function to scale the coord grid to something visible
	const MUL = 50.0
	const OFF = 10.0
	scale := func(x float64) float64 { return MUL*x + OFF }

	svgElem := svgElement{
		Width:    float64(dims[1])*MUL + 2*OFF,
		Height:   float64(dims[0])*MUL + 2*OFF,
		Polygons: make([]interface{}, rows*2),
	}
	for row := 0; row < rows; row++ {
		coord := coords.RowView(row)
		var colorMask []int
		classID, classFound := classes[row]
		// if no class information, just use shades of gray
		if !classFound || classID == -1 {
			colorMask = []int{255, 255, 255}
		} else {
			colorMask = colors[classes[row]%len(colors)]
		}
		colorMul := 1.0 - (umatrix[row]-minDistance)/(maxDistance-minDistance)
		r := int(colorMul * float64(colorMask[0]))
		g := int(colorMul * float64(colorMask[1]))
		b := int(colorMul * float64(colorMask[2]))
		polygonCoords := ""
		x := scale(coord.At(0, 0))
		y := scale(coord.At(1, 0))
		// hexagon has a different yOffset
		switch uShape {
		case "hexagon":
			{
				xOffset := 0.5 * MUL
				yBigOffset := math.Tan(math.Pi/6.0) * MUL
				ySmallOffset := yBigOffset / 2.0
				// draw a hexagon around the current coord
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+ySmallOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x, y+yBigOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y+ySmallOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y-ySmallOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x, y-yBigOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y-ySmallOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+ySmallOffset)
			}
		default:
			{
				xOffset := 0.5 * MUL
				yOffset := 0.5 * MUL
				// draw a box around the current coord
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y-yOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y-yOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y+yOffset)
				polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)
			}
		}

		svgElem.Polygons[row*2] = polygon{
			Points: []byte(polygonCoords),
			Style:  fmt.Sprintf("fill:rgb(%d,%d,%d);stroke:black;stroke-width:1", r, g, b),
		}

		// print class number
		if classFound {
			svgElem.Polygons[row*2+1] = textElement{
				X:    x - 0.25*MUL,
				Y:    y + 0.25*MUL,
				Text: fmt.Sprintf("%d", classes[row]),
			}
		}
	}

	elems = append(elems, svgElem)

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
