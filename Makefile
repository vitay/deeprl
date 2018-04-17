all: html

pdf: DeepRL.md
	pandoc -s -F pandoc-crossref --toc  \
		--csl=apalike.csl -V fontsize=11pt -V roboto:sfdefault -V geometry:margin=1in \
		-V documentclass=report -V linestretch=1.3 DeepRL.md -o DeepRL.tex
	rubber --pdf DeepRL.tex
	rubber --clean DeepRL.tex

html: DeepRL.md
	pandoc -N -F pandoc-crossref -F pandoc-citeproc --template=default.html5 --mathjax="/usr/share/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML" \
		--toc --listings --css=github.css \
		--csl=apalike.csl DeepRL.md -o DeepRL.html

export: DeepRL.md
	pandoc -N -F pandoc-crossref --template=default.html5 --mathjax="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML" \
		-V fontpath="<link href="https://fonts.googleapis.com/css?family=Roboto" rel="stylesheet">" \
		--toc --listings --css=github.css \
		--csl=apalike.csl DeepRL.md -o index.html


