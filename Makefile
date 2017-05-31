all: pdf html

pdf: DeepRL.md
	pandoc -sSN -F pandoc-crossref --toc --listings --bibliography=/home/vitay/Articles/biblio/ReinforcementLearning.bib --csl=apalike.csl -V fontsize=11pt -V fontfamily=roboto -V geometry:margin=1in -V documentclass=report -V linestretch=1.3 DeepRL.md -o DeepRL.tex
	rubber --pdf DeepRL.tex
	rubber --clean DeepRL.tex

html: DeepRL.md
	pandoc -sSN -F pandoc-crossref --template=default.html5 --mathjax="/usr/share/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"  --toc --listings --css=github.css --bibliography=/home/vitay/Articles/biblio/\
ReinforcementLearning.bib --csl=apalike.csl DeepRL.md -o DeepRL.html


