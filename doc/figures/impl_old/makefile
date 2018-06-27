
FIG = fig.fig
EPS = fig.eps
IMG = fig.jpeg

DOC = doc.ps
DVI = doc.dvi
TEX = doc.tex

figure: $(IMG)

$(IMG): $(DOC)
	convert -density 400 -resize 3000x $(DOC) -background White -flatten $(IMG)

$(DOC): $(TEX) $(EPS)
	latex $(TEX)
	dvips $(DVI)

$(EPS): $(FIG)
	fig2dev -L eps -m 2.00 -c $(FIG) $(EPS)

clean:
	rm -f $(DVI) $(DOC) $(EPS) $(IMG)

.PHONY:
	clean
