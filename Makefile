
all: ulysses-figure-labels

install: all
	cp ulysses-figure-labels /usr/local/bin

ulysses-figure-labels: ulysses-figure-labels.hs
	ghc $<

