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
				 python/rocmlm.py
# Cleanup directories
DATAPURGE = log \
						python/__pycache__ \
						$(DATADIR)/synth*.csv \
						$(DATADIR)/bench-pca.csv \
						$(DATADIR)/gfem_summaries \
						$(DATADIR)/earthchem-pca.csv \
						$(DATADIR)/earthchem-counts.csv \
						$(DATADIR)/temp-dataset.parquet \
DATACLEAN = assets Perple_X gfems pretrained_rocmlms
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) gfems rocmlms

datasets: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) -u python/datasets.py $(LOG)
	@echo "=============================================" $(LOG)

test: $(LOGFILE) $(PYTHON) pca
	@$(CONDAPYTHON) -u python/test.py $(LOG)
	@echo "=============================================" $(LOG)

rocmlms: $(LOGFILE) $(PYTHON) pca
	@PYTHONWARNINGS="ignore" $(CONDAPYTHON) -u python/rocmlm.py $(LOG)
	@echo "=============================================" $(LOG)

gfems: pca
	@$(CONDAPYTHON) -u python/gfem.py $(LOG)

pca: initialize
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

.PHONY: clean purge find_conda_env create_conda_env remove_conda_env get_assets initialize pca gfems rocmlms all