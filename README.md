[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FabeG/assess_ntc_with_pypowsybl/HEAD?labpath=assess_ntc_with_pypowsybl.ipynb)

This repository contains a jupyter notebook showing how the &DeltaNTC; methodology has been implemented in the TYNDP to calculate additional Transfer Capacity brought by a new interconnection project.

It also illustrates how easily [pypowsybl](https://github.com/powsybl/pypowsybl) can be used to perform all the steps involving network manipulation or network calculation.

# Installation

- If you want to run the notebook on your computer, you will have to install all the needed packages, just run:

   ``pip install -r requirements.txt``

- You can also run a version of the notebook with all the packages already installed by just clicking on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FabeG/assess_ntc_with_pypowsybl/HEAD?labpath=assess_ntc_with_pypowsybl.ipynb) (but can take some time, please be patient...)

# Presentation of the notebook

This notebook illustrates all the steps involved in &Delta;NTC computation:

- import of CGMES individual grid models,
- merge these models into a common grid model,
- calculate PTDF matrix by running DC load flows,
- calculate LODF matrix,
- calculate powershift sensitivity,
- calculate HVDC sensitivity,
- extract ratings from the CGMES models,
- and calculate maximum powershift 

With this approach, it will be possible to quickly calculate yearly flows (on hourly basis, which means 8760 hours) in N condition.

By using a solver (Google OR-Tool), it will be possible to calculate yearly &Delta;NTC, in N/N-1 condition, optimizing an HVDC to increase the Transfer Capacity.
