
TEXFLAGS ?= -file-line-error -halt-on-error -interaction=nonstopmode

FIGS = figures/vape_all figures/vape_pe figures/vape_update figures/vape_prediction figures/vope_all figures/vope_pe figures/vope_update figures/vope_prediction

BIBFILE = main.bib
MAIN = main

default: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).ps
	ps2pdf $< $@

$(MAIN).ps: $(MAIN).dvi
	dvips -o $@ $<

$(MAIN).dvi: $(MAIN).tex figs
	latex $(MAIN)
	bibtex $(MAIN).aux
	latex $(MAIN)
	latex $(MAIN)

figs:
	@for figDir in $(FIGS); do make -C $$figDir; done

clean:
	$(RM) $(MAIN).{pdf,ps,dvi,aux,log}
	@for figDir in $(FIGS); do make -C $$figDir clean; done

.PHONY: clean figs
