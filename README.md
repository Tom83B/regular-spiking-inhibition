# regular-spiking-inhibition

This repository contains the scripts and Jupyter Notebooks we used to obtain the data and generate the figures for our manuscript Regular spiking in high conductance states: the essential role of inhibition, currently available at [arxiv](https://arxiv.org/abs/2101.08731).

Repositary structure
* `LIF` - scripts used for generating data from GLIF models simulations
  * `abstract.py` - script for generating the subthreshold statistics data in the Graphical Abstract (Fig. 1B,C)
  * `obtain_lif_heatmap.py` - script for obtaining the values of CV for various combinations of PSFR x E-I ratio or PSFR x presynaptic inhibitory activity
* `HH` - scripts used for generating data from HH models simulations
  * `abstract.py` - script for generating the data for the Hodgkin-Huxley type models in the Graphical Abstract (Fig. 1D,E)
  * `obtain_HH_heatmap.py` - equivalent of `obtain_lif_heatmap.py` for the HH type models
* `utils.py` - scipt containing helpful fuctions and classes used throughout the repository
* `refine_clines.py` - script used for refining the results obtained from `LIF/obtain_lif_heatmap.py` and `HH/obtain_HH_heatmap.py`, i.e., increasing point density and simulation duration where needed, for Figs. 3,4
* `refine_const_inh.py` - same as `refine_clines.py` for Fig. 5
* `abstract.ipynb` - Jupyter Notebook that generates the Graphical Abstract (Fig. 1)
* `heatmap figures.ipynb` - Jupyter Notebook that generates Figs. 3,4
* `constant inhibition.ipynb` - Jupyter Notebook that generates Fig. 5
