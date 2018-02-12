
PDF = doc.pdf
TEX = doc.tex

$(PDF): $(TEX) *.tex
	pdflatex $(TEX)

clean:
	rm -f *.aux *.log $(PDF)

.PHONY: clean
