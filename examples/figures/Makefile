all: python1.pdf python2.pdf haskell.pdf

python1.pdf: divide-and-conquer/index.md ../../python/py1-ulysses-figure-labels
		pandoc divide-and-conquer/index.md --resource-path=divide-and-conquer -F ../../python/py1-ulysses-figure-labels -F pandoc-crossref -o python1.pdf

python2.pdf: divide-and-conquer/index.md ../../python/py2-ulysses-figure-labels
			pandoc divide-and-conquer/index.md --resource-path=divide-and-conquer -F ../../python/py2-ulysses-figure-labels -F pandoc-crossref -o python2.pdf

haskell.pdf: divide-and-conquer/index.md ../../haskell/ulysses-figure-labels
	pandoc divide-and-conquer/index.md --resource-path=divide-and-conquer -F ../../haskell/ulysses-figure-labels -F pandoc-crossref -o haskell.pdf
