package dataset

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// load data funcs
var loadFuncs = map[string]func(io.Reader) (*mat64.Dense, error){
	".csv": LoadCSV,
}

// DataSet represents training data set
type DataSet struct {
	data *mat64.Dense
}

// New returns new data set or fails with error if either the path to data set
// supplied as a parameter does not exist or if the file is encoded
// in an unsupported format. File format is inferred from the file extension.
// Currently only csv files are supported.
func New(path string) (*DataSet, error) {
	// Check if the supplied file type is supported
	fileType := filepath.Ext(path)
	loadData, ok := loadFuncs[fileType]
	if !ok {
		return nil, fmt.Errorf("Unsupported file type: %s\n", fileType)
	}
	// Check if the training data file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, err
	}
	// Open training data file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// Load file
	data, err := loadData(file)
	if err != nil {
		return nil, err
	}
	// Return Data
	return &DataSet{
		data: data,
	}, nil
}

// Data returns the data stored in a matrix
func (ds DataSet) Data() *mat64.Dense {
	return ds.data
}

// Scale normalizes data in each column based on its mean and standard deviation and returns it.
// It modifies the underlying daata. If this is not desirable use the standalone Scale function.
func (ds *DataSet) Scale() *mat64.Dense {
	return scale(ds.data, true)
}

// LoadCSV loads data set from the path supplied as a parameter.
// It returns data matrix that contains particular CSV fields in columns.
// It returns error if the supplied data set contains corrrupted data or
// if the data can not be converted to float numbers
func LoadCSV(r io.Reader) (*mat64.Dense, error) {
	// data matrix dimensions: rows x cols
	var rows, cols int
	// mxData contains ALL data read field by field
	var mxData []float64
	// create new CSV reader
	csvReader := csv.NewReader(r)
	// read all data record by record
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		// allocate the dataRow during the first iteration
		if rows == 0 {
			// initialize cols on first iteration
			cols = len(record)
		}
		// convert strings to floats
		for _, field := range record {
			f, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, err
			}
			// append the read data into mxData
			mxData = append(mxData, f)
		}
		rows++
	}
	// return data matrix
	return mat64.NewDense(rows, cols, mxData), nil
}

// Scale centers the data set to zero mean values in each column and then normalizes them.
// It does not modify the data stored in the matrix supplied as a parameter.
func Scale(mx mat64.Matrix) *mat64.Dense {
	return scale(mx, false)
}

// scale centers the supplied data set to zero mean in each column and then normalizes them.
// You can specify whether you want to scale data in place or return new data set
func scale(mx mat64.Matrix, inPlace bool) *mat64.Dense {
	rows, cols := mx.Dims()
	// mean/stdev store each column mean/stdev values
	col := make([]float64, rows)
	mean := make([]float64, cols)
	stdev := make([]float64, cols)
	// calculate mean and standard deviation for each column
	for i := 0; i < cols; i++ {
		// copy i-th column to col
		mat64.Col(col, i, mx)
		mean[i], stdev[i] = stat.MeanStdDev(col, nil)
	}
	// initialize scale function
	scale := func(i, j int, x float64) float64 {
		return (x - mean[j]) / stdev[j]
	}
	// if in place data should be modified
	if inPlace {
		mxDense := mx.(*mat64.Dense)
		mxDense.Apply(scale, mxDense)
		return mxDense
	}
	// otherwise allocate new data matrix
	dataMx := new(mat64.Dense)
	dataMx.Clone(mx)
	dataMx.Apply(scale, dataMx)
	return dataMx
}
