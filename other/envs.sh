conda create -n mashtree -c conda-forge mamba
conda activate mashtree
mamba install -c bioconda mashtree

conda create -n mybase -c conda-forge mamba
conda activate mybase
mamba install -c conda-forge -c bioconda jupyterlab matplotlib pandas pathlib  markov_clustering biopython networkx sklearn-pandas pyvips pillow pyyaml

conda create -n easyfig -c conda-forge python=2.7 pillow pyvips

