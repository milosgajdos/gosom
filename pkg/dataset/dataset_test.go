package dataset

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

var (
	fileName = "test.csv"
)

func setup() {
	// create a correct test file
	content := []byte("2.0,3.5\n4.5,5.5\n7.0,9.0")
	tmpPath := filepath.Join(os.TempDir(), fileName)
	if err := ioutil.WriteFile(tmpPath, content, 0666); err != nil {
		log.Fatal(err)
	}
}

func teardown() {
	// remove test file
	os.Remove(filepath.Join(os.TempDir(), fileName))
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// delete test files
	teardown()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestDataSet(t *testing.T) {
	assert := assert.New(t)

	tmpPath := path.Join(os.TempDir(), fileName)
	// create new dataset
	ds, err := New(tmpPath, false)
	assert.NoError(err)
	assert.NotNil(ds)

	// retrieve data and check dimensions
	mx := ds.Data()
	rows, cols := mx.Dims()
	assert.Equal(3, rows)
	assert.Equal(2, cols)

	// pre-computed test data
	scaled := []float64{
		-1, -0.8980265101338746,
		0, -0.1796053020267749,
		1, 1.0776318121606494,
	}
	scaledMx := mat64.NewDense(3, 2, scaled)
	// scale data set in place
	scaledDs := ds.Scale()
	assert.True(mat64.Equal(scaledMx, scaledDs))
	assert.True(mat64.Equal(scaledMx, ds.Data()))

	// Unsupported file format
	ds, err = New("example", false)
	assert.Error(err)

	// Nonexistent file
	ds, err = New(path.Join(".", "nonexistent.csv"), false)
	assert.Error(err)
}

func TestDataWithClasses(t *testing.T) {
	assert := assert.New(t)

	fileName := "TestDataWithClasses"

	// create a .lrn file
	lrnPath := path.Join(os.TempDir(), fileName+".lrn")
	lrn := `% 4
% 4
% 9	1	1	1	
% Key	C1	C2	C3	
1	0.000000E+000	0.000000E+000	1.000000E+000
2	0.000000E+000	5.233600E-002	9.986300E-001
3	4.977400E-002	1.617300E-002	9.986300E-001
4	3.076200E-002	-4.234100E-002	9.986300E-001
`
	if err := ioutil.WriteFile(lrnPath, []byte(lrn), 0666); err != nil {
		log.Fatal(err)
	}

	// create a corresponding .cls file - only some rows have classification information
	clsPath := path.Join(os.TempDir(), fileName+".cls")
	cls := `% 2
1	1
4	2
`
	if err := ioutil.WriteFile(clsPath, []byte(cls), 0666); err != nil {
		log.Fatal(err)
	}

	ds, err := New(lrnPath, true)
	assert.NoError(err)
	assert.NotNil(ds)
	rows, cols := ds.Data().Dims()
	assert.Equal(4, rows)
	assert.Equal(3, cols)
	assert.Equal(2, len(ds.Classes()))
	// index is 0-based so '4' becomes '3' here
	assert.Equal(2, ds.Classes()[3])
}

func TestLoadCSV(t *testing.T) {
	assert := assert.New(t)

	// correct data
	tstRdr := strings.NewReader("1,2,3")
	mx, err := LoadCSV(tstRdr)
	assert.NoError(err)
	r, c := mx.Dims()
	assert.Equal(r, 1)
	assert.Equal(c, 3)

	// inconsisten data
	tstRdr = strings.NewReader("1,2,3\n4,5")
	mx, err = LoadCSV(tstRdr)
	assert.Error(err)
	assert.Nil(mx)

	// corrupted data i.e. can't convert to float
	tstRdr = strings.NewReader("1,sdfsdfd,3\n4,5")
	mx, err = LoadCSV(tstRdr)
	assert.Error(err)
	assert.Nil(mx)
}

func TestLoadLRN(t *testing.T) {
	assert := assert.New(t)

	// simple
	tstRdr := strings.NewReader(`# some comment
% 1
% 2
% 9	1
% Key	Value
1	1`)
	mx, err := LoadLRN(tstRdr)
	assert.NoError(err)
	assert.NotNil(mx)
	rows, cols := mx.Dims()
	assert.Equal(1, rows)
	assert.Equal(1, cols)

	// real-like
	tstRdr = strings.NewReader(`% 4
% 4
% 9	1	1	1	
% Key	C1	C2	C3	
1	0.000000E+000	0.000000E+000	1.000000E+000
2	0.000000E+000	5.233600E-002	9.986300E-001
3	4.977400E-002	1.617300E-002	9.986300E-001
4	3.076200E-002	-4.234100E-002	9.986300E-001
`)
	mx, err = LoadLRN(tstRdr)
	assert.NoError(err)
	assert.NotNil(mx)

	// invalid header - one header row missing
	tstRdr = strings.NewReader(`% 1
% 2
% 9	1
1	1
`)
	mx, err = LoadLRN(tstRdr)
	assert.Equal("Invalid header", err.Error())
	assert.Nil(mx)

}

func TestLoadCLS(t *testing.T) {
	assert := assert.New(t)

	// simple
	tstRdr := strings.NewReader(`# some comment
% 3
1	1
2	2
3	3
`)
	cls, err := LoadCLS(tstRdr)
	assert.NoError(err)
	assert.NotNil(cls)
	assert.Equal(3, len(cls))
	assert.Equal(1, cls[0])

	// unsupported header (this is permitted by the original .cls format)
	tstRdr = strings.NewReader(`# some comment
% 3
% 1 class1	255	0	0
% 2 class2	0	255	0
% 3 class3	0	0	255
1	1
2	2
3	3
`)
	cls, err = LoadCLS(tstRdr)
	assert.Error(err)
	assert.Nil(cls)
	assert.Equal("Unsupported header", err.Error())

	// invalid header
	tstRdr = strings.NewReader(`# some comment
1	1
2	2
3	3
`)
	cls, err = LoadCLS(tstRdr)
	assert.Error(err)
	assert.Nil(cls)
	assert.Equal("Invalid header", err.Error())
}

func TestScale(t *testing.T) {
	assert := assert.New(t)

	// unlabeled data set
	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := New(tmpPath, false)
	assert.NoError(err)
	assert.NotNil(ds)
	// pre-computed test data
	scaled := []float64{
		-1, -0.8980265101338746,
		0, -0.1796053020267749,
		1, 1.0776318121606494,
	}
	scaledMx := mat64.NewDense(3, 2, scaled)
	scaledDs := Scale(ds.Data())
	assert.True(mat64.Equal(scaledDs, scaledMx))
}
