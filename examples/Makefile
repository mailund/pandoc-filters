all: ulysses-fig-goal.pdf ulysses-fig.pdf

ulysses-fig-goal.pdf:
	pandoc ulysses-fig-goal.md -F pandoc-crossref -o ulysses-fig-goal.pdf
	
ulysses-fig.pdf:
	pandoc ulysses-fig.md -F ../ulysses-figure-labels -F pandoc-crossref -o ulysses-fig-goal.pdf
	