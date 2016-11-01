BINARY=gosom
BUILD=go build
CLEAN=go clean
INSTALL=go install
BUILDPATH=./_build
PACKAGES=$(shell go list ./... | grep -v /vendor/)

build: builddir
	$(BUILD) -v -o $(BUILDPATH)/gosom

all: builddir build

install:
	$(INSTALL) ./...
clean:
	rm -rf $(BUILDPATH)
	rm -rf $(GOPATH)/bin/$(BINARY)
builddir:
	mkdir -p $(BUILDPATH)
test:
	for pkg in ${PACKAGES}; do \
		go vet $$pkg || exit ; \
		golint $$pkg || exit ; \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean build
