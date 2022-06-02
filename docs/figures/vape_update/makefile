
DESTDIR ?= .

SRC = figure.skel.tex
SVG = $(wildcard *.svg)

EPS = $(dir $(DESTDIR)/)$(patsubst %.svg,%.eps,$(SVG))
EPS_TEX = $(dir $(DESTDIR)/)$(patsubst %.svg,%.eps_tex,$(SVG))
TEX = $(dir $(DESTDIR)/)figure.tex

INKSCAPE = inkscape
FLAGS = -z --export-ps-level=3 --export-dpi 500 --export-latex

default: $(TEX) $(EPS)

$(TEX): $(SRC) $(EPS)
	sed '/^ *% *INKSCAPE/q' $< > $@
	cat $(EPS_TEX) >> $@
	sed '1,/^ *% *INKSCAPE/d' $< >> $@

$(EPS): $(SVG) | $(DESTDIR)
	$(INKSCAPE) $(FLAGS) --export-eps=$@ $<
	if test -n "$$BLACK_AND_WHITE"; then \
	convert -density 400 $@ -quality 100 -size 2000x -type Grayscale .tmp.jpg && \
	sam2p .tmp.jpg EPS: $@ && $(RM) .tmp.jpg; \
	fi

$(DESTDIR):
	mkdir -p $(DESTDIR)

clean:
	$(RM) $(EPS) $(EPS_TEX) $(TEX)

.PHONY: clean
