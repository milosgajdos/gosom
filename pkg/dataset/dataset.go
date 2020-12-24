package dataset

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// LRN data format constants
const (
	// LrnHeaderSize holds the size of LRN file header
	LrnHeaderSize = iota
	// LrnHeaderCols holds number of LRN header columns
	LrnHeaderCols
	// LrnHeaderTypes holds number of LRN types
	LrnHeaderTypes
	// LrnHeaderNames not used
	LrnHeaderNames
	// LrnHeaderRows holds number of LRN header rows
	LrnHeaderRows
)

// load data funcs
var loadFuncs = map[string]func(io.Reader) (*mat.Dense, error){
	".csv": LoadCSV,
	".lrn": LoadLRN,
}

// load classifications funcs
var loadClsFuncs = map[string]func(io.Reader) (map[int]int, error){
	".cls": LoadCLS,
}

// DataSet represents training data set
type DataSet struct {
	Data    *mat.Dense
	Classes map[int]int
}

// New returns pointer to dataset or fails with error if either the file
// in dataPath does not exist or if it is encoded in an unsupported format.
// File format is inferred from the file extension. Currently only csv and lrn
// data formats are supported.
// If the dataset has classification information it can be provided as the second
// parameter. If the file in clsPath doesn't exist New fails with error.
func New(dataPath string, clsPath string) (*DataSet, error) {
	// Check if the supplied file type is supported
	fileType := filepath.Ext(dataPath)
	loadData, ok := loadFuncs[fileType]
	if !ok {
		return nil, fmt.Errorf("unsupported file type: %s", fileType)
	}
	// Check if the training data file exists
	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		return nil, err
	}
	// Open training data file
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// Load file
	data, err := loadData(file)
	if err != nil {
		return nil, err
	}
	// Load classes
	classes := make(map[int]int) // default empty classification information
	if clsPath != "" {
		// Check if the classification file type is supported
		clsFileType := filepath.Ext(clsPath)
		loadCls, ok := loadClsFuncs[clsFileType]
		if !ok {
			return nil, fmt.Errorf("unsupported type of classification file: %s", clsFileType)
		}
		// Check if the classification file exists
		if _, err := os.Stat(clsPath); os.IsNotExist(err) {
			return nil, err
		}
		clsFile, err := os.Open(clsPath)
		if err != nil {
			return nil, err
		}
		defer clsFile.Close()
		classes, err = loadCls(clsFile)

		if err != nil {
			return nil, err
		}
	}
	// Return Data
	return &DataSet{
		Data:    data,
		Classes: classes,
	}, nil
}

// Scale normalizes data in each column based on its mean and standard deviation and returns it.
// It modifies the underlying daata. If this is not desirable use the standalone Scale function.
func (ds *DataSet) Scale() *mat.Dense {
	return scale(ds.Data, true)
}

// LoadCSV loads data set from the path supplied as a parameter.
// It returns data matrix that contains particular CSV fields in columns.
// It returns error if the supplied data set contains corrrupted data or
// if the data can not be converted to float numbers
func LoadCSV(r io.Reader) (*mat.Dense, error) {
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
	return mat.NewDense(rows, cols, mxData), nil
}

// LoadLRN reads data from a .lrn file.
// See the specification here: http://databionic-esom.sourceforge.net/user.html#Data_files____lrn_
func LoadLRN(reader io.Reader) (*mat.Dense, error) {
	const DataCol = 1
	var rows, cols int
	var mxData []float64
	headerRow := 0
	columnTypes := []int{}
	valueRow := 0

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\t ")
		if strings.HasPrefix(line, "#") { // comment
			continue
		} else if strings.HasPrefix(line, "%") { // header
			headerLine := strings.TrimPrefix(line, "% ")
			if headerRow == LrnHeaderSize { // rows
				rows64, err := strconv.ParseInt(headerLine, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("dataset size information missing")
				}
				rows = int(rows64)
			} else if headerRow == LrnHeaderCols { // cols
				// discard
			} else if headerRow == LrnHeaderTypes { // col types
				colTypes := strings.Split(headerLine, "\t")
				for _, colType := range colTypes {
					ct, err := strconv.ParseInt(colType, 10, 64)
					if err != nil {
						return nil, err
					}
					columnTypes = append(columnTypes, int(ct))
					// we're interested in data columns only
					if ct == DataCol {
						cols++
					}
				}
				// allocate data matrix because we know rows and cols now
				mxData = make([]float64, rows*cols)
			}
			headerRow++
		} else { // data
			if headerRow < LrnHeaderRows {
				return nil, fmt.Errorf("invalid header")
			}
			if valueRow >= rows {
				return nil, fmt.Errorf("too many data rows")
			}
			vals := strings.Split(line, "\t")
			valueIndex := 0
			for i, val := range vals {
				if i > len(columnTypes) {
					return nil, fmt.Errorf("too many columns")
				}
				if columnTypes[i] == DataCol {
					if valueIndex < cols {
						f, err := strconv.ParseFloat(val, 64)
						if err != nil {
							return nil, fmt.Errorf("problem parsing value at line %d, col %d", valueRow, i)
						}
						mxData[valueRow*cols+valueIndex] = f
						valueIndex++
						continue
					}
					return nil, fmt.Errorf("too many data columns")
				}
			}
			valueRow++
		}
	}
	if valueRow != rows {
		return nil, fmt.Errorf("Wrong number of data rows.  Expecting %d, but was %d", rows, valueRow)
	}

	return mat.NewDense(rows, cols, mxData), nil
}

// LoadCLS reads classification information from a .cls file.
// See the specification here: http://databionic-esom.sourceforge.net/user.html#Classification_files____cls_
// The only supported header is the Number of datasets (n)
func LoadCLS(reader io.Reader) (map[int]int, error) {
	var rows *int
	valueRow := 0
	classifications := make(map[int]int)

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\t ")
		if strings.HasPrefix(line, "#") { // comment
			continue
		} else if strings.HasPrefix(line, "%") { // header
			if rows != nil {
				return nil, fmt.Errorf("unsupported header")
			}
			headerLine := strings.TrimPrefix(line, "% ")
			rows64, err := strconv.ParseInt(headerLine, 10, 64)
			if err != nil {
				fmt.Println(err)
				return nil, fmt.Errorf("classification data size information missing")
			}
			rowsTmp := int(rows64)
			rows = &rowsTmp
		} else { // classes
			if rows == nil {
				return nil, fmt.Errorf("invalid header")
			}
			if valueRow >= *rows {
				return nil, fmt.Errorf("too many classification rows")
			}
			vals := strings.Split(line, "\t")
			// classes come in pairs: index -> class
			var index, class *int
			for i, val := range vals {
				if i > 1 {
					return nil, fmt.Errorf("too many classification columns")
				}
				num64, err := strconv.ParseInt(val, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("problem parsing value at line %d, col %d", valueRow, i)
				}
				num := int(num64)

				if i == 0 {
					index = &num
				} else if i == 1 {
					class = &num
				}
			}
			if index == nil || class == nil {
				return nil, fmt.Errorf("incomplete classification row")
			}

			// CLS indexes are 1-based, but we're using 0-based
			classifications[*index-1] = *class
			valueRow++
		}
	}
	return classifications, nil
}

// Scale centers the data set to zero mean values in each column and then normalizes them.
// It does not modify the data stored in the matrix supplied as a parameter.
func Scale(mx mat.Matrix) *mat.Dense {
	return scale(mx, false)
}

// scale centers the supplied data set to zero mean in each column and then normalizes them.
// You can specify whether you want to scale data in place or return new data set
func scale(mx mat.Matrix, inPlace bool) *mat.Dense {
	rows, cols := mx.Dims()
	// mean/stdev store each column mean/stdev values
	col := make([]float64, rows)
	mean := make([]float64, cols)
	stdev := make([]float64, cols)
	// calculate mean and standard deviation for each column
	for i := 0; i < cols; i++ {
		// copy i-th column to col
		mat.Col(col, i, mx)
		mean[i], stdev[i] = stat.MeanStdDev(col, nil)
	}
	// initialize scale function
	scale := func(i, j int, x float64) float64 {
		return (x - mean[j]) / stdev[j]
	}
	// if in place data should be modified
	if inPlace {
		mxDense := mx.(*mat.Dense)
		mxDense.Apply(scale, mxDense)
		return mxDense
	}
	// otherwise allocate new data matrix
	dataMx := new(mat.Dense)
	dataMx.CloneFrom(mx)
	dataMx.Apply(scale, dataMx)
	return dataMx
}
