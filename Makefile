
all: html

html:
	rm -f *.html
	python make.py
	ln -s Introduction.html index.html

pdf: html
	pandoc -sN --toc --top-level-division=chapter -F pandoc-crossref -F pandoc-citeproc --metadata=crossrefYaml:"assets/pandoc-crossref.yaml" document.md -o DeepRL.pdf


