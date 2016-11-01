# gosom

[![GoDoc](https://godoc.org/github.com/milosgajdos83/gosom?status.svg)](https://godoc.org/github.com/milosgajdos83/gosom)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/gosom.svg?branch=master)](https://travis-ci.org/milosgajdos83/gosom)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/gosom)](https://goreportcard.com/report/github.com/milosgajdos83/gosom)
[![codecov](https://codecov.io/gh/milosgajdos83/gosom/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/gosom)

This project provides an implementation of [Self-Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM) in Go. In addition the project provides few useful packages that can be used independently in your other projects.

The project provides an implementation of both `sequential` and `batch` mode SOM training algorithms. You're encourage to read up on both before you start using the program as the choice of available command line options can have a noticeable effect on performance and resulting output of the training.

# Get started

Get the source code:

```
$ go get -u github.com/milosgajdos83/gosom
```

Run the tests:

```
$ make test
```

Build and install in `$GOPATH/bin`:

```
make install
```

See the `Makefile` for more available `make` tasks.

Once the program has successfully built you can inspect available command line options it provides:

```
$ gosom -h
```

# Examples

Try to run the following examples using the `FCPS` dataset (see below). Change the `$D` environment variable to test different datasets. Both programs output an `HTML` file with [U-matrix](https://en.wikipedia.org/wiki/U-matrix) rendered as `SVG` image.

## Batch algorithm

```
$ D=Target; go build; ./gosom -umxout umatrix_batch.html -dims 30,30 -radius 500.0 -rdecay exp -ushape rectangle -iters 100 -training batch -input testdata/fcps/${D}.lrn -clsinput testdata/fcps/${D}.cls
```

## Sequential algorithm

```
D=Hepta; go build; ./gosom -umxout umatrix_seq.html -dims 30,30 -radius 500.0 -rdecay exp -lrate 0.5 -ldecay exp -ushape rectangle -iters 30000 -training seq -input testdata/fcps/${D}.lrn -clsinput testdata/fcps/${D}.cls
```

# Acknowledgements

Test data present in `fcps` subdirectory of `testdata` come from [Philipps University of Marburg](http://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1):

**Ultsch, A.**: Clustering with SOM: U*C, In *Proc. Workshop on Self-Organizing Maps, Paris, France, (2005) , pp. 75-82*
