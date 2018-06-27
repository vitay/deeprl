
all: html

html:
	python make.py

pdf: html
	pandoc -sN --toc --top-level-division=chapter -F pandoc-crossref -F pandoc-citeproc --metadata=crossrefYaml:"assets/pandoc-crossref.yaml" document.md -o document.pdf


