# gosom

[![GoDoc](https://godoc.org/github.com/milosgajdos83/gosom?status.svg)](https://godoc.org/github.com/milosgajdos83/gosom)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/gosom.svg?branch=master)](https://travis-ci.org/milosgajdos83/gosom)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/gosom)](https://goreportcard.com/report/github.com/milosgajdos83/gosom)
[![codecov](https://codecov.io/gh/milosgajdos83/gosom/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/gosom)

**THIS PROJECT IS A WIP!!!**

This project will provide an implementation of [Self-Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM) in Go.

# Examples
Try the following simple examples using the FCPS dataset (see below):

```
# Batch algorithm ()
export P=Target; go build; ./gosom -umxout umatrix_batch.html -dims 30,30 -radius 500.0 -rdecay exp -ushape rectangle -iters 100 -training batch -input testdata/fcps/${P}.lrn -clsinput testdata/fcps/${P}.cls

# Sequential algorithm
export P=Hepta; go build; ./gosom -umxout umatrix_seq.html -dims 30,30 -radius 500.0 -rdecay exp -lrate 0.5 -ldecay exp -ushape rectangle -iters 30000 -training seq -input testdata/fcps/${P}.lrn -clsinput testdata/fcps/${P}.cls
```
Change the $P variable for different data sets

# Acknowledgements

Test data present in `fcps` subdirectory of `testdata` come from [Philipps University of Marburg](http://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1):

**Ultsch, A.**: Clustering with SOM: U*C, In *Proc. Workshop on Self-Organizing Maps, Paris, France, (2005) , pp. 75-82*
