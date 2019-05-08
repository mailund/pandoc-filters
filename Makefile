
all: ulysses-figure-labels #conditionals

install: all
	cp haskell/ulysses-figure-labels /usr/local/bin
	cp python/py[12]-ulysses-figure-labels /usr/local/bin
	cp exercises.py /usr/local/bin

ulysses-figure-labels: haskell/ulysses-figure-labels.hs
	ghc $<
