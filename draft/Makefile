MS = kerswell-et-al-rocmlm2
SUP = kerswell-et-al-rocmlm2-supporting

all: ms sup

ms: $(MS).pdf
	@open $<

sup: $(SUP).pdf
	@open $<

$(MS).pdf: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(MS).tex: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(SUP).pdf: $(SUP).Rmd
	@R -q -e 'rmarkdown::render("$(SUP).Rmd", quiet=T, clean=F)'

$(SUP).tex: $(SUP).Rmd
	@R -q -e 'rmarkdown::render("$(SUP).Rmd", quiet=T, clean=F)'

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk} *.knit.md $(MS).tex $(SUP).tex $(MS).pdf $(SUP).pdf

clean: purge

.PHONY: clean purge get_tables sup ms all