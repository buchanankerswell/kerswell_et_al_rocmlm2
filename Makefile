# Logging config
DATE = $(shell date +"%d-%m-%Y")
LOGFILE := log/log-$(DATE)
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda config
CONDAENVNAME = rocmlm2
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Directories with data and perplex configs
DATADIR = assets
# Python scripts
PYTHON = python/pca.py \
				 python/gfem.py \
				 python/utils.py \
				 python/rocmlm.py \
				 python/write-md-tables.py
# Cleanup directories
DATAPURGE = log \
						python/__pycache__ \
						$(DATADIR)/synth*.csv \
						$(DATADIR)/bench-pca.csv \
						$(DATADIR)/gfem_summaries \
						$(DATADIR)/earthchem-pca.csv \
						$(DATADIR)/lut-efficiency.csv \
						$(DATADIR)/earthchem-counts.csv \
						$(DATADIR)/gfem-model-results-summary.csv
DATACLEAN = assets Perple_X gfems rocmlms python/HyMaTZ
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) gfems rocmlms

write_md_tables: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) -u python/write-md-tables.py $(LOG)

test: $(LOGFILE) $(PYTHON) mixing_arrays
	@$(CONDAPYTHON) -u python/test.py $(LOG)
	@echo "=============================================" $(LOG)

rocmlms: $(LOGFILE) $(PYTHON) mixing_arrays
	@PYTHONWARNINGS="ignore" $(CONDAPYTHON) -u python/rocmlm.py $(LOG)
	@echo "=============================================" $(LOG)

gfems: mixing_arrays
	@$(CONDAPYTHON) -u python/gfem.py $(LOG)

mixing_arrays: initialize
	@$(CONDAPYTHON) -u python/pca.py $(LOG)

initialize: $(LOGFILE) $(PYTHON) create_conda_env get_assets

get_assets: $(DATADIR)

$(DATADIR): $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) -u python/utils.py $(LOG)

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p log; \
		touch $(LOGFILE); \
	fi

remove_conda_env:
	@echo "Removing conda env $(CONDAENVNAME) ..."
	@conda remove --name $(CONDAENVNAME) --all --yes

create_conda_env: $(LOGFILE) $(CONDASPECSFILE) find_conda_env
	@if [ "$(HASCONDA)" = "false" ]; then \
		echo "Install conda first!" $(LOG); \
		echo "See: https://github.com/buchanankerswell/kerswell_et_al_rocmlm_hydrated" $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MYENVDIR)" ]; then \
		echo "Conda environment \"$(CONDAENVNAME)\" found!" $(LOG); \
	else \
		echo "Creating conda environment $(CONDAENVNAME) ..." $(LOG); \
		conda env create --file $(CONDASPECSFILE) $(LOG) > /dev/null 2>&1; \
		echo "Conda environment $(CONDAENVNAME) created!" $(LOG); \
	fi

find_conda_env: $(LOGFILE)
	$(eval MYENVDIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)

.PHONY: clean purge find_conda_env create_conda_env remove_conda_env get_assets mixing_arrays gfems rocmlms initialize write_md_tables all