all: html

html:
	cp /Users/vitay/Articles/bibtex/DeepLearning.bib .
	cp /Users/vitay/Articles/bibtex/ReinforcementLearning.bib .
	quarto render .