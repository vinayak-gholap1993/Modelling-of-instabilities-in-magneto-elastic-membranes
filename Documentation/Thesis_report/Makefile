all: dvi pdf ps

pdf: complete.pdf

ps: complete.ps

dvi: complete.dvi

fastdvi: complete.tex task.tex abstract.tex chapter*.tex appendix*.tex references.bib
	latex complete.tex

rebuild: distclean all

clean:
	rm -f *.log *.aux *.toc *.lof *.bbl 
	rm -f *.blg *.idx *.ind *.ilg *.glo *.gls
	rm -f *.lot *.loa

distclean: clean
	rm -f *.dvi *.pdf *.ps
	rm -f *~

mrproper: distclean

complete.pdf: complete.tex abstract.tex chapter*.tex appendix*.tex references.bib
	pdflatex complete
	bibtex complete
	makeindex complete.idx -s complete.ist -o complete.ind
	makeindex complete.glo -s nomencl.ist -o complete.gls
	pdflatex complete
	pdflatex complete

complete.ps: complete.dvi
	dvips -o complete.ps complete.dvi

complete.dvi: complete.tex abstract.tex chapter*.tex appendix*.tex references.bib
	latex complete
	bibtex complete
	makeindex complete.idx -s complete.ist -o complete.ind
	makeindex complete.glo -s nomencl.ist -o complete.gls
	latex complete
	latex complete
