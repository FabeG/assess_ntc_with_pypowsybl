[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FabeG/assess_ntc_with_pypowsybl/HEAD?labpath=assess_ntc_with_pypowsybl.ipynb)

This repository contains a jupyter notebook showing how the &DeltaNTC; methodology has been implemented in the TYNDP to calculate additional Transfer Capacity brought by a new interconnection project.

It also illustrates how easily [pyPowSyBl](https://github.com/powsybl/pypowsybl) can be used to perform all the steps involving network manipulation or network calculation.

# Installation

- If you want to run the notebook on your computer, you will have to install all the needed packages, just run:

   ``pip install -r requirements.txt``

- You can also run a version of the notebook with all the packages already installed by just clicking on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FabeG/assess_ntc_with_pypowsybl/HEAD?labpath=assess_ntc_with_pypowsybl.ipynb) (but can take some time, please be patient...)

# Presentation of the notebook

In this notebook, we will use the [pyPowSyBl](https://github.com/powsybl/pypowsybl) package to cover all the steps involved in calculating transfer capacity for electricity interconnections:

- short presentation of the pyPowsybl library that will be used in each step of this workshop,
- import the FR, ES and PT network models (from the public TYNDP2022 dataset provided by Entsoe),
- merge these 3 models to generate a common grid model,
- use OpenData and pyPowsybl to draw part of the merged network,
- calculate PTDF matrix (sensitivities of lines to Load / Generation / interconections / ...),
- thanks to Entsoe Transparency Platform, get yearly data (Load / Generation / Crossborder exchanges) on hourly basis,
- calculate yearly flows on the common grid network,
- calculate the sensitivity to a powershift between France and Spain,
- extract the ratings of monitored lines from the model,
- calculate the transfer capacity in N condition,
- calcultate OTDF matrix (to take into account impact of N-1 on other lines),
- calculate the transfer capacity in N-1 condition,
- create the Baixas - StLlogaia HVDC in the model,
- calculate the sensitivity of lines to this HVDC,
- calculate the transfer capacity in N-1 condition using the HVDC to maximize it.

copyright RTE France - licence MPL 2.0
