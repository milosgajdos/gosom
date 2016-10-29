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
	ds, err := New(tmpPath)
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
	ds, err = New("example")
	assert.Error(err)

	// Nonexistent file
	ds, err = New(path.Join(".", "nonexistent.csv"))
	assert.Error(err)
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

func TestLRN(t *testing.T) {
	assert := assert.New(t)

	tstRdr := strings.NewReader("# some comment\n% 1\n% 1\n% 9\t1\n% Key\tValue\n1\t1")
	mx, err := LoadLRN(tstRdr)
	assert.NoError(err)
	assert.NotNil(mx)
}

func TestScale(t *testing.T) {
	assert := assert.New(t)

	// unlabeled data set
	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := New(tmpPath)
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
