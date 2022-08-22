
FIG = fig.fig
EPS = fig.eps
IMG = fig.jpeg
EPS2 = fig2.eps

DOC = doc.ps
DVI = doc.dvi
TEX = doc.tex

figure: $(EPS2)

$(EPS2): $(IMG)
	convert $(IMG) $(EPS2)

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
