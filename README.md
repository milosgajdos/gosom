# gosom

[![GoDoc](https://godoc.org/github.com/milosgajdos83/gosom?status.svg)](https://godoc.org/github.com/milosgajdos83/gosom)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/gosom.svg?branch=master)](https://travis-ci.org/milosgajdos83/gosom)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/gosom)](https://goreportcard.com/report/github.com/milosgajdos83/gosom)
[![codecov](https://codecov.io/gh/milosgajdos83/gosom/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/gosom)

This project provides an implementation of [Self-Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM) in Go. It implements the two most well known SOM training algorithms: `sequential` and `batch`. The `batch` training algorithm is faster than the `sequential` as it can be parallelized, taking advantage of as many cores as your machine provides, however it can be less accurate. `Batch` training provides a resonable approximation of SOM and thus its results can be less accurate than the ones produced by `sequential` algorithm, but still acceptable. The `sequential` algorithm is performed as its name implies, sequentially and therefore it's slower, but more accurate. You can find more information about SOM training algorithms [here](http://www.scholarpedia.org/article/Kohonen_network).

The goal of the project is to provide a simple API to build SOM in `Go`. Apart from the SOM API build packages, the project` also implements various SOM quality measures which can help you validate the results of the training algorithm. In particular the project implements `quantization` and `topographic` error to measure both the projection and topography as well as `topographic product` which can help you make a decision about the size of the SOM grid.

# Get started

Get the source code:

```
$ go get -u github.com/milosgajdos83/gosom
```

Run the tests:

```
$ make test
```

# Example

You can see the simplest example of `SOM` below:

```go

```

To get you started quickly you can run the examples below. We will use [FCPS dataset]((http://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1)). Change the `$D` environment variable to play with different example datasets. Both examples below output an `HTML` file with [U-matrix](https://en.wikipedia.org/wiki/U-matrix) rendered as `SVG` image.

## Batch algorithm

```
$ D=Target; go run main.go -umatrix umatrix_batch.html -dims 30,30 -radius 500.0 -rdecay exp -ushape rectangle -iters 100 -training batch -input testdata/fcps/${D}.lrn -cls testdata/fcps/${D}.cls
[ gosom ] Loading data set testdata/fcps/Target.lrn
[ gosom ] Creating new SOM. Dimensions: [30 30], Grid: planar, Unit shape: rectangle
[ gosom ] Starting SOM training. Method: batch, iterations: 100
[ gosom ] Training successfully completed. Duration: 1.923548243s
[ gosom ] Saving U-Matrix to umatrix_batch.html
[ gosom ] Quantization Error: 0.023212
[ gosom ] Topographic Product: +Inf
[ gosom ] Topographic Error: 0.015584
```

## Sequential algorithm

```
$ D=Target; go run main.go -umatrix umatrix_seq.html -dims 30,30 -radius 500.0 -rdecay exp -lrate 0.5 -ldecay exp -ushape hexagon -iters 30000 -training seq -input testdata/fcps/${D}.lrn -cls testdata/fcps/${D}.cls
[ gosom ] Loading data set testdata/fcps/Target.lrn
[ gosom ] Creating new SOM. Dimensions: [30 30], Grid: planar, Unit shape: hexagon
[ gosom ] Starting SOM training. Method: seq, iterations: 30000
[ gosom ] Training successfully completed. Duration: 2.261068582s
[ gosom ] Saving U-Matrix to umatrix_seq.html
[ gosom ] Quantization Error: 0.064704
[ gosom ] Topographic Product: 0.010281
[ gosom ] Topographic Error: 0.014286
```
# Acknowledgements

Test data present in `fcps` subdirectory of `testdata` come from [Philipps University of Marburg](http://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1):

**Ultsch, A.**: Clustering with SOM: U*C, In *Proc. Workshop on Self-Organizing Maps, Paris, France, (2005) , pp. 75-82*
