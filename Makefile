
all: ulysses-figure-labels #conditionals

install: all
	cp ulysses-figure-labels /usr/local/bin
	cp py[12]-ulysses-figure-labels /usr/local/bin

ulysses-figure-labels: ulysses-figure-labels.hs
	ghc $<

conditionals: conditionals.hs
		ghc $<
