EXDIR=examples
BINARY=gosom
BUILD=go build
CLEAN=go clean
INSTALL=go install
PACKAGES=$(shell go list ./... | grep -v /vendor/ | grep -v /examples/)
EXAMPLES=$(wildcard examples/*/*.go)

build: $(EXAMPLES)
	$(BUILD) -v $(EXAMPLES)

all: build

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
