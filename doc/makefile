
TEXFLAGS ?= -file-line-error -halt-on-error -interaction=nonstopmode

FIGS = figures/vape_all figures/vape_pe figures/vape_update figures/vape_prediction figures/vope_all figures/zoom_vape_with_vope figures/zoom_vope figures/diffeq_vape_without_nu figures/diffeq_vape figures/diffeq_vope 

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
