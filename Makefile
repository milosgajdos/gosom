BUILD=go build
CLEAN=go clean
INSTALL=go install
BUILDPATH=./_build
PACKAGES=$(shell go list ./... | grep -v /vendor/ | grep -v /examples/)
EXAMPLES=$(wildcard examples/*/*.go)

examples: builddir
	for example in $(EXAMPLES); do \
		go build $$example ; \
	done

all: examples

builddir:
	mkdir -p $(BUILDPATH)

install:
	$(INSTALL) ./$(EXDIR)/...

clean:
	rm -rf $(BUILDPATH)

check:
	for pkg in ${PACKAGES}; do \
		go vet $$pkg || exit ; \
		golint $$pkg || exit ; \
	done

test:
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean build
