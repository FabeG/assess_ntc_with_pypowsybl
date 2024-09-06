#!/usr/bin/env python
# coding: utf-8

# # Using pyPowSyBl to assess transfer capacity of new electricity interconnections

# https://github.com/FabeG/assess_ntc_with_pypowsybl

# ## Getting started
# 
# ### Either by running the notebook from binder
# 
# You can run the notebook remotely: no python nor packages installation needed, just click on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FabeG/assess_ntc_with_pypowsybl/HEAD?labpath=assess_ntc_with_pypowsybl.ipynb)
# 
# ### Or downloading the repository from github
# 
# If you prefer runnning the notebook on your machine:
# 
# - you should have ```python``` installed on your laptop (**tested with python 3.10**)
# - you can install all the required python packages as usual:
# 
# ```pip install -r requirements.txt```

# Github repository can be found at https://github.com/FabeG/assess_ntc_with_pypowsybl

# ## Introduction

# Assessing the benefits of interconnection projects is one of the goals of ENTSO-E Ten-Year Network Development Plan (TYNDP).
# 
# New electricity interconnections, by increasing power exchange capabilities between countries are key elements to make the energy transition happen in a cost effective and secure way.
# 
# Being able to evaluate the increased power flows brought by a new interconnection project is then of major importance.
# To calculate this indicator (named ΔNTC: increase in Net Transfer Capacity), a tool chain based on [pyPowSyBl](https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/) (from Powsybl Linux Fundation Energy project) has been developed within an ENTSO-E working group and will be used for the next TYNDP.
# 
# The methodology regarding &Delta;NTC calculation is in line with the [Implementation Guidelines for TYNDP2024](https://tyndp.entsoe.eu/resources/tyndp-2024-implementation-guidelines) (except for some simplifications)
# 
# Based on a real HVDC interconnection project between France and Spain and publicly available data, we will illustrate the methodology used and all pyPowSyBl fonctionnalities involved through a Jupyter notebook:
# 
# - Importing TYNDP2022 French, Spanish and Portuguese transmission grid model (in CGMES format) and merging them to build a common grid model,
# - Calculating yearly load flow on hourly basis,
# - Taking into account the impact of line tripping,
# - Calculating the ΔNTC brought by the interconnection project.

# ## Notebook presentation
# 
# In this notebook, we will use the pyPowsybl package to cover all the steps involved in calculating transfer capacity for electricity interconnections:
# 
# - short presentation of the [pyPowsybl library](https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/) that will be used in each step of this workshop,
# - import the FR, ES and PT network models (from the public TYNDP2022 dataset provided by Entsoe),
# - merge these 3 models to generate a common grid model,
# - use OpenData and pyPowsybl to draw part of the merged network,
# - calculate PTDF matrix (sensitivities of lines to Load / Generation / interconections / ...),
# - thanks to Entsoe Transparency Platform, get yearly data (Load / Generation / Crossborder exchanges) on hourly basis,
# - calculate yearly flows on the common grid network,
# - calculate the sensitivity to a powershift between France and Spain,
# - extract the ratings of monitored lines from the model,
# - calculate the transfer capacity in N condition,
# - calcultate OTDF matrix (to take into account impact of N-1 on other lines),
# - calculate the transfer capacity in N-1 condition,
# - create the Baixas - StLlogaia HVDC in the model,
# - calculate the sensitivity of lines to this HVDC,
# - calculate the transfer capacity in N-1 condition using the HVDC to maximize it.
# 
# ## Global workflow
# 
# ![title](./images/workflow.png)

# In[1]:


import io
import math
import re
from pathlib import Path
from collections import defaultdict

from tqdm.notebook import tqdm
from IPython.display import SVG
import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
import plotly.graph_objects as go
import pypowsybl as pp

import warnings
warnings.filterwarnings("ignore")

pd.options.plotting.backend = "plotly"

try:
    import entsoe_secrets
    proxies = entsoe_secrets.proxies
except ImportError:
    # To get a token to use Entsoe RESTful API:
    # - first, register at https://transparency.entsoe.eu
    # - once registered, send an email to transparency@entsoe.eu with “Restful API access” in the subject line.
    #   Indicate the email address you entered during registration in the email body. 
    token = "put you token here"
    # If you are behind a corporate proxy, you have to define the following dict
    # proxies = {
    #     "http": "http://your_login:your_passwd@ip_address_of_proxy:port_number",
    #     "https": "http://your_login:your_passwd@ip_address_of_proxy:port_number"
    # }
    proxies = {}
colors = ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF"]


# ---
# ## Quick tour on how to use pyPowSybl
# 
# 
# **Note**: pyPowSyBl documentation can be found at https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/index.html
# 
# 
# Once a network model is loaded in pyPowSybl, all the elements of the network can be accessed by using the following methods :
# 
# | Element | method |
# |----|----|
# | voltage levels  | [``network.get_voltage_levels()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_voltage_levels.html#pypowsybl.network.Network.get_voltage_levels) |
# | substations | [``network.get_substations()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_substations.html#pypowsybl.network.Network.get_substations) |
# | buses | [``network.get_buses()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_buses.html#pypowsybl.network.Network.get_buses) |
# | loads | [``network.get_loads()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_loads.html#pypowsybl.network.Network.get_loads) |
# | generators | [``network.get_generators()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_generators.html#pypowsybl.network.Network.get_generators) |
# | lines | [``network.get_lines()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_lines.html#pypowsybl.network.Network.get_lines) |
# | 2 windings transformers | [``network.get_2_windings_transformers()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_2_windings_transformers.html#pypowsybl.network.Network.get_2_windings_transformers) |
# | xinjections | [``network.get_dangling_lines()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_dangling_lines.html#pypowsybl.network.Network.get_dangling_lines) |
# | tie-lines | [``network.get_tie_lines()``](https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/reference/api/pypowsybl.network.Network.get_tie_lines.html#pypowsybl.network.Network.get_tie_lines) |
# 
# For a complete description of all the elements of the network and how to interact with them, you can refer to [pyPowSybl documentation](https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/network.html#reading-network-elements-data)
# 

# ### Load a test case MicroGrid

# In[2]:


network_test = pp.network.create_micro_grid_be_network()


# ### Draw the whole network

# In[3]:


network_test.get_network_area_diagram(
    [],
    nad_parameters=pp.network.NadParameters(
        edge_name_displayed=False,
        substation_description_displayed=True,
        edge_info_along_edge=True
    )
)


# ### Substations

# ```mermaid
#     erDiagram
#         substations {
#             id rdfid PK
#             string name
#             string CGMESregionName
#             string geo_tags
#         }
# ```

# In[4]:


substations = network_test.get_substations(attributes=["name", "CGMES.regionName", "geo_tags"])
substations


# In[5]:


network_test.get_single_line_diagram("87f7002b-056f-4a6a-a872-1744eea757e3", pp.network.SldParameters(use_name=True))


# In[6]:


network_test.get_single_line_diagram("37e14a0f-5e34-4647-a062-8bfd9305fa9d", pp.network.SldParameters(use_name=True))


# ### Voltage levels
# ```mermaid
#     erDiagram
#         voltage_levels {
#             id rdfid PK
#             string name
#             float nominal_v
#             string substation_id FK
#         }
# ```

# In[7]:


voltage_levels = network_test.get_voltage_levels().sort_values(by="substation_id")
voltage_levels


# In[8]:


voltage_levels[["name", "substation_id", "nominal_v"]].merge(substations, left_on="substation_id", right_index=True, suffixes=("", "_subst"))


# ### Loads
# 
# ```mermaid
#     erDiagram
#         loads {
#             id rdfid PK
#             float p0
#             string bus_id FK
#             string voltage_level_id FK
#         }
# ```

# In[9]:


network_test.get_loads()


# ### Generators
# ```mermaid
#     erDiagram
#         generators {
#             id rdfid PK
#             string name
#             string energy_source
#             float target_p
#             float max_p
#             string bus_id FK
#             string voltage_level_id FK
#         }
# ```

# In[10]:


network_test.get_generators()


# ### External injections
# ```mermaid
#     erDiagram
#         dangling_lines {
#             id rdfid PK
#             string name
#             float p0
#             string voltage_level_id FK
#         }
# ```

# In[11]:


network_test.get_dangling_lines()


# ### Buses
# 
# ```mermaid
#     erDiagram
#         buses {
#             id rdfid PK
#             int synchronous_component
#         }
# ```

# In[12]:


network_test.get_buses()


# ## Using real network models from TYNDP public dataset
# 
# Individual grid models from previous TYNDP can be requested at https://www.entsoe.eu/publications/statistics-and-data/#entso-e-on-line-application-portal-for-network-datasets: 
# 
# ![title](./images/tyndp_dataset.png)

# ### Load the French, Spanish and Portuguese networks using pyPowsyBl

# In[13]:


boundary_set = "20211216T1459Z_ENTSO-E_BD_1346.zip"
parameters = {
    # the boundary set will be imported
    "iidm.import.cgmes.boundary-location": str((Path("./data") / boundary_set).resolve()),
    # rdfID will be used as id for elements
    "iidm.import.cgmes.source-for-iidm-id": "rdfID",
}
models = {
    "ES": "ES_3PQT.zip",
    "FR": "FR_3PQT_v3.zip",
    "PT": "PT_3PQT_v2.zip",
}
networks = {}
for tso, model in tqdm(models.items()):
    networks[tso] = pp.network.load(Path("./data") / model, parameters=parameters)


# ### Merge ES, PT and FR models

# In[14]:


network = networks["FR"]
network.merge([networks["ES"], networks["PT"]])


# ### Generate useful dataframes

# #### For voltage_levels
# 
# ```mermaid
#     erDiagram
#         voltage_levels ||--|| substations : substation_id
#         voltage_levels {
#             id rdfid PK
#             string name
#             string substation_id FK
#             float nominal_v
#         }
#         substations {
#             id rdfid PK
#             string name
#             string CGMESregionName
#             string geo_tags
#         }
# ```

# In[15]:


voltage_levels = network.get_voltage_levels(attributes=["name", "substation_id", "nominal_v"])
voltage_levels.head()


# In[16]:


substations = network.get_substations(attributes=["name", "CGMES.regionName", "geo_tags"])
substations.head()


# In[17]:


voltage_levels = voltage_levels.merge(substations, left_on="substation_id", right_index=True, suffixes=("", "_subst"))
voltage_levels.head()


# In[18]:


buses = network.get_buses(attributes=["synchronous_component"])


# #### For generators
# 
# ```mermaid
#     erDiagram
#         generators ||--|| buses : bus_id
#         generators ||--|| voltage_levels : voltage_level_id
#         voltage_levels ||--|| substations : substation_id
#         generators {
#             id rdfid PK
#             string name
#             string energy_source
#             float target_p
#             float max_p
#             string bus_id FK
#             string voltage_level_id FK
#         }
#         buses {
#             id rdfid PK
#             int synchronous_component
#             string voltage_level_id
#         }
#         voltage_levels {
#             id rdfid PK
#             float nominal_v
#             string substation_id FK
#         }
#         substations {
#             id rdfid PK
#             string name
#             string CGMESregionName
#             string geo_tags
#         }
# ```

# In[19]:


generators = network.get_generators(attributes=["name", "energy_source", "target_p", "max_p", "bus_id", "voltage_level_id"]) 
generators = generators.merge(buses, left_on="bus_id", right_index=True, how="left")
generators = generators.merge(voltage_levels, left_on="voltage_level_id", right_index=True, suffixes=("", "_vl"))
generators = generators[generators["synchronous_component"] == 0]
generators.head()


# #### For loads
# 
# ```mermaid
#     erDiagram
#         loads ||--|| buses : bus_id
#         loads ||--|| voltage_levels : voltage_level_id
#         voltage_levels ||--|| substations : substation_id
#         loads {
#             id rdfid PK
#             float p0
#             string bus_id FK
#             string voltage_level_id FK
#         }
#         buses {
#             id rdfid PK
#             int synchronous_component
#             string voltage_level_id
#         }
#         voltage_levels {
#             id rdfid PK
#             float nominal_v
#             string substation_id FK
#         }
#         substations {
#             id rdfid PK
#             string name
#             string CGMESregionName
#             string geo_tags
#         }
# ```
# 

# In[20]:


loads = network.get_loads(attributes=["p0", "bus_id", "voltage_level_id"])
loads = loads.merge(buses, left_on="bus_id", right_index=True, how="left")
loads = loads.merge(voltage_levels, left_on="voltage_level_id", right_index=True, suffixes=("", "_vl"))
loads = loads[loads["synchronous_component"] == 0]
loads.head()


# #### Lines

# In[21]:


lines = network.get_lines(attributes=["name", "p1", "connected1", "connected2"])
lines


# #### Dangling lines / External injections
# 
# To get external injections from the models, we will use `dangling_lines` that represents half of a tie-line.
# 
# For a more detailed explanation, on how dangling lines are modelized, you can have a look at the [Powsybl documentation](https://powsybl.readthedocs.io/projects/powsybl-core/en/latest/grid_model/network_subnetwork.html#dangling-line)

# In[22]:


xinjections = network.get_dangling_lines(attributes=["name", "p0", "p", "voltage_level_id", "tie_line_id"])
xinjections.head()


# ##### Merge with voltage levels to retrieve nominal_v

# In[23]:


xinjections = (
    xinjections
    .merge(voltage_levels["nominal_v"], left_on="voltage_level_id", right_index=True, how="left")
)
xinjections.head()


# #### Tie-lines

# In[24]:


tie_lines = network.get_tie_lines()
tie_lines.head()


# In[25]:


tie_lines = (
    tie_lines
    # merge with xinjections to retrieve nominal voltages on both sides
    .merge(xinjections[["nominal_v", "p"]], left_on="dangling_line1_id", right_index=True, how="left")
    .merge(xinjections[["nominal_v", "p"]], left_on="dangling_line2_id", right_index=True, how="left", suffixes=("1", "2"))
)
tie_lines


# #### Overall figures

# In[26]:


result = {
    "substations": len(substations),
    "voltage_levels": len(voltage_levels),
    "buses": len(buses),
    "generators": len(generators),
    "loads": len(loads),
    "lines": len(lines),
    "tie-lines": len(tie_lines),
    "xinjections": len(xinjections[xinjections["tie_line_id"] == ""])
}
pd.Series(result)


# In[27]:


def display_voltage_level(voltage_level_name):
    vl = network.get_voltage_levels(attributes=["name"])
    if voltage_level_name in vl.index:
        vl_id = voltage_level_name
    else:
        vl_id = vl[vl["name"] == voltage_level_name].index[0]
    return network.get_single_line_diagram(
        vl_id,
        pp.network.SldParameters(
            use_name=True,
            diagonal_label=True,
            topological_coloring=True,
            nodes_infos=False,
            tooltip_enabled=True,
        )
    )


# ## Geographical representation of the network near the France - Spain border

# ### Using OpenData to get RTE substations coordinates
# 
# The website ODRE (https://odre.opendatasoft.com/explore) present lots of OpenData related to RTE.
# 
# We will use here the dataset related to RTE substations coordinates: https://odre.opendatasoft.com/explore/dataset/postes-electriques-rte/table/?disjunctive.fonction&disjunctive.etat&disjunctive.tension&sort=-code_poste
# 
# ![title](./images/substations_coordinates.png)

# #### Read the csv file containing RTE substations coordinates

# In[28]:


gps_coords_fr = (
    pd.read_csv(
        Path("./data") / "postes-electriques-rte.csv",
        sep=";",
        encoding="latin1",
        usecols=["Code poste", "Longitude poste (DD)", "Latitude poste (DD)"]
    )
    .rename(columns={"Longitude poste (DD)": "longitude", "Latitude poste (DD)": "latitude"})
    .set_index("Code poste")
)
# Unknown coordinates for BEHLA substation : put it arbitratry in the middle of the line CANTEL71SAUCA (and little longitude deviation to better see the double circuit)
gps_coords_fr.loc["BEHLA"] = (gps_coords_fr.loc["CANTE"] + gps_coords_fr.loc["SAUCA"]) / 2
gps_coords_fr.loc["BEHLA", "longitude"] += 0.1
gps_coords_fr


# #### Filter substations to show (only south-west of France / voltage levels higher than 220 kV / no fictitious substations)

# In[29]:


vl_for_gps_fr = (
    voltage_levels[
        # Only south-west french network
        (voltage_levels["geo_tags"] == "RTE-TOULOUSE")
        # 220 kV and 400 kV
        & (voltage_levels["nominal_v"] >= 220)
        # don't take into account fictitious substations 
        & (~voltage_levels["name"].str.startswith(("1", "2", "3")))
    ]
)
subst_for_gps_fr = vl_for_gps_fr.set_index("substation_id")["name"].str[:-2].str.strip().to_frame()


# In[30]:


coords_fr = (
    gps_coords_fr
    .merge(subst_for_gps_fr.reset_index(), left_on="Code poste", right_on="name")
    .set_index("substation_id")
)
coords_fr.head()


# ### Using geocoder service to find GPS coordinates of Red Electrica substations

# #### Filter substations to show (only spanish substations close to the border / voltage levels higher than 220 kV)

# In[31]:


idx_es = network.get_network_area_diagram_displayed_voltage_levels(
    list(voltage_levels[voltage_levels["name"].isin(["HERNANI", "BIESCAS", "VIC", "LLOGAIA"])].index),
    depth=2
)
subst_for_gps_es =  (
    voltage_levels
    .loc[idx_es]
    .loc[(voltage_levels["CGMES.regionName"] == "ES") & (voltage_levels["nominal_v"] > 200)][["substation_id", "name"]]
    .drop_duplicates(subset=["substation_id"])
    .set_index("substation_id")["name"].to_frame()
)
subst_for_gps_es.head()


# In[32]:


# Hhelp Nominatim to find the city names
city = {
    "LAFARGA": "JUIA",
    "RIUDAREN": "RIUDARENES",
    "SABINANI": "SABINANIGO",
    "GATICA": "GATIKA",
    "LAROCA": "LA ROCA VILLAGE",
    "SENTMENA": "SENTMENAT",
    "TESCALON": "ESCALONA ARAGON",
    "FRANQUES": "FRANQUESES",
    "PALAU": "PALAU-SOLITA",
    "CASTEJON": "CASTEJON DE EBRO",
    "ABANTO": "ABANTO Y",
    "AMOREBIE": "AMOREBIETA",
}


# #### Find spanish substations coordinates using geocoder (return GPS coordinates based on city names)

# In[33]:


gps_coords_es = {}
to_execute = True
if to_execute:
    from geopy.geocoders import Nominatim
    import time
    geolocator = Nominatim(user_agent="ntc_with_pypowsybl", proxies=proxies)
    for _, subst in tqdm(subst_for_gps_es.drop_duplicates(subset="name").iterrows()):
        location = geolocator.geocode(query={"city": city.get(subst["name"],subst["name"]) }, country_codes="es")
        if not location:
            location = geolocator.geocode(subst["name"],country_codes="es")
        try:
            gps_coords_es[subst["name"]] = (location.latitude, location.longitude)
        except:
            pass
        time.sleep(1.1)


# In[34]:


coords_es = (
    subst_for_gps_es
    .merge(pd.DataFrame.from_dict(gps_coords_es, orient="index", columns=["latitude", "longitude"]), left_on="name", right_index=True)
)
coords_es.head()


# #### Merge spanish and french substations coordinates

# In[35]:


coords_fr_es = pd.concat([coords_fr, coords_es]).rename_axis("id")[["latitude", "longitude"]]
coords_fr_es.head()


# #### Update ``substationsPosition`` extension (enrich substations to add GPS coordinates in pyPowSyBl)

# In[36]:


network.remove_extensions('substationPosition', network.get_extensions('substationPosition').index)
network.create_extensions('substationPosition', coords_fr_es[["latitude", "longitude"]])


# ### Show the GeoGraphical network map

# In[37]:


# Don't want to see disconnected lines
lines = network.get_lines(["connected1", "connected2"])
network.remove_elements(lines[~lines["connected1"] | ~lines["connected2"]].index)
try:
    # Show the Geographical map
    from pypowsybl_jupyter import NetworkMapWidget
    mapview = NetworkMapWidget(network, nominal_voltages_top_tiers_filter=5, use_name=True)
except ImportError:
    parameter = pp.network.NadParameters(layout_type=pp.network.NadLayoutType.GEOGRAPHICAL, bus_legend=False)
    mapview = SVG(network.get_network_area_diagram(list(vl_for_gps_fr.index.union(vl_for_gps_es.index)), nad_parameters=parameter).svg.replace("25px", "250px").replace("stroke-width: 5", "stroke-width: 15"))
mapview


# ## Network modification

# ### Remove generators modelizing HVDC injections

# In[38]:


hvdc = generators["name"].str.contains("VSC") | generators["name"].str.contains("HVDC")
generators[hvdc]


# In[39]:


network.remove_elements(generators[hvdc].index)
generators = generators.loc[~hvdc]


# ## Calculate sensitivity of lines to parameters : PTDF matrix

# #### Calculate initial flows using DC load flow

# In[40]:


def get_flows(network):
    # Get lines active power flow
    lines = network.get_lines(attributes=["name", "p1"])
    # Get tie-line active power flow
    ext_injections = network.get_dangling_lines(attributes=["name", "p"])
    tie_lines = network.get_tie_lines(attributes=["name", "dangling_line1_id", "dangling_line2_id"])
    tie_lines = (
        tie_lines
        .merge(ext_injections["p"], left_on="dangling_line1_id", right_index=True, how="left")
        .merge(ext_injections["p"], left_on="dangling_line2_id", right_index=True, how="left", suffixes=("1", "2"))
    )
    flows = pd.concat([lines, tie_lines])
    return flows.reset_index().set_index(["id", "name"])["p1"]


# In[41]:


# Run DC load flow
parameters_lf = pp.loadflow.Parameters(
    distributed_slack=False,
    connected_component_mode=pp.loadflow.ConnectedComponentMode.MAIN,
)
res = pp.loadflow.run_dc(network, parameters=parameters_lf)
# Get flows on all lines
initial_flows = get_flows(network)

# Build a dict between line_names and id
lines_id_dict = initial_flows.reset_index(level=1)["name"].to_dict()
print(res)
print("")
print(f"There is an unbalance of {res[0].slack_bus_results[0].active_power_mismatch:.1f} MW")
print("(too much generation started compared to the load)")


# #### First sensitivity calculation: example with load increase in Spain

# ##### Increase the load in Spain by 100 MW

# In[42]:


# Define load_increase (in MW)
LOAD_INCREASE = 100
# selection of loads in Spain
load_es = loads[loads["CGMES.regionName"] == "ES"].copy()
# Increase the load in Spain (proportionnaly to each load)
load_es["p0"] = load_es["p0"] + LOAD_INCREASE * load_es["p0"] / load_es["p0"].sum()
# Update the network
network.update_loads(load_es[["p0"]])


# ##### Perform new load flow after load increase

# In[43]:


res = pp.loadflow.run_dc(network, parameters=parameters_lf)
flows_after_load_increase = get_flows(network)

print(f"There is an unbalance of {res[0].slack_bus_results[0].active_power_mismatch:.1f} MW")
print(f"(100 MW less than in the previous load flow: normal since we increased the load by {LOAD_INCREASE} MW)")


# ##### Calculate sensitivity to load increase in Spain

# In[44]:


sensitivity = (flows_after_load_increase - initial_flows) / 100
sensitivity.dropna().sort_values()


# ##### Go back to initial values

# In[45]:


network.update_loads(loads.loc[loads["CGMES.regionName"] == "ES", ["p0"]])
res = pp.loadflow.run_dc(network, parameters=parameters_lf)
print(f"Unbalance is {res[0].slack_bus_results[0].active_power_mismatch:.1f} MW: OK, same as initial load flow")


# #### Parameters selection

# In[46]:


parameters = {
    "ES_Wind Onshore":
        {"type": "generators", "filter": (generators["CGMES.regionName"] == "ES") & (generators["energy_source"] == "WIND")},
    "ES_Other_Gen":
        {"type": "generators", "filter": (generators["CGMES.regionName"] == "ES") & (generators["energy_source"] != "WIND")},
    "ES_Load":
        {"type": "loads", "filter": loads["CGMES.regionName"] == "ES"},
    "FR_Nuclear":
        {"type": "generators", "filter": (generators["CGMES.regionName"] == "FR") & (generators["energy_source"] == "NUCLEAR")},
    "FR_Other_Gen":
        {"type": "generators", "filter": (generators["CGMES.regionName"] == "FR") & (generators["energy_source"] != "NUCLEAR")},
    "FR_Load": 
        {"type": "loads", "filter": loads["CGMES.regionName"] == "FR"},
    "PT_Gen_Total":
        {"type": "generators", "filter": generators["CGMES.regionName"] == "PT00"},
    "PT_Load": 
        {"type": "loads", "filter": loads["CGMES.regionName"] == "PT00"},
    "FR_BE":
        {"type": "exchange", "filter": xinjections["name"].isin([".AVELL71MASTA", ".ACHEL71LONNY", ".AVELL72AVELI", ".AUBAL61M.MA5", ".AUBAL61MOULA", "MOULAL61SOTEL", ".FMONL61CHOO5" ])},
    "FR_IT": 
        {"type": "exchange", "filter": xinjections["name"].isin([".VENAL71VLARO", ".RODPL71ALBER", ".RODPL72ALBER"])},
    "FR_UK":
        {"type": "exchange", "filter": xinjections["name"].isin(["BIPOLL72MANDA", ".IFA2L71TOURB", "BIPOLL71MANDA", "MANDAL71PEUP5", ])},
    "FR_DE":
        {"type": "exchange", "filter": xinjections["name"].isin([".EICL72MUHL_PINT228", ".ENSDL71VIGY", ".ENSDL72VIGY", ".EICL73MUHL_PINT228", ])},
    "FR_CH":
        {"type": "exchange", "filter": xinjections["name"].isin(["PRESSL61VALLO", ".VERBL71B.TOL", ".SSTRL61CORNI", ".ROMAL71B.TOL", ".VERBL61GEN.P", ".VERBL62GEN.P", ".BASSL71SIERE", ".BASSL71MAMBE", ".ASPHL71SIERE", ".LAUFL71SIERE", ".RIDDL61CORNI" ])},
    "ES_MA": 
        {"type": "exchange", "filter": xinjections["name"].isin(["TARIFA_XTA_FA12_2_400", "TARIFA_XTA_FA11_1_400"])},
    "FR_IE": 
        {"type": "exchange", "filter": xinjections["name"].isin([".CELTICL71MARTY"])},
}


# <div class="alert alert-block alert-info">
# For the sake of simplicity, we chose to select only a few parameters: 
#     
# - regarding generation per type of production we only considered separately **Nuclear units in France** and **Wind in Spain**: in usual calculations, there is more granularity on sensitivities calculated per type of production
# - **no PST** sensitivity (we consider the PST on their neutral tap position in all the calculations
# - no separation between **industrial loads** (NonConformLoad) and **residential load** (ConformLoad)
# </div>
# 

# #### Initial values of the selected parameters

# In[47]:


initial_tso_data = {}
for nom, param in parameters.items():
    if param["type"] == "generators":
        initial_tso_data[nom] = generators.loc[param["filter"], "target_p"].sum()
    elif param["type"] == "loads":
        initial_tso_data[nom] = loads.loc[param["filter"], "p0"].sum()
    elif param["type"] == "exchange":
        initial_tso_data[nom] = xinjections.loc[param["filter"], "p0"].sum()
initial_tso_data = pd.Series(initial_tso_data)
initial_tso_data


# #### Check that balance is correct
# 
# The balance is defined as the sum of all the generators - loads - crossborder_exchanges_out

# In[48]:


balance = (
    initial_tso_data[["ES_Wind Onshore", "ES_Other_Gen", "FR_Nuclear", "FR_Other_Gen", "PT_Gen_Total"]].sum()
    - initial_tso_data[["FR_BE", "FR_IT", "FR_UK", "FR_DE", "FR_CH", "ES_MA", "FR_IE"]].sum()
    - initial_tso_data[["FR_Load", "ES_Load", "PT_Load"]].sum()
)
balance


# In line with the unbalance calculated by the DC load flow:

# In[49]:


print(f"{res[0].slack_bus_results[0].active_power_mismatch:.1f}")


# #### Calculating sensitivities to all selected parameters

# In[50]:


sensitivity = {}
for nom, param in parameters.items():
    if param["type"] == "generators":
        # Select the generators we want to calculate sensitivity for
        gen_changed = generators[parameters[nom]["filter"]].copy()
        # Increase the selected generators by 100 MW (proportionnaly to the max_p of each generator) 
        gen_changed["target_p"] = gen_changed["target_p"] + 100 * gen_changed["max_p"] / gen_changed["max_p"].sum()
        # Update the network with the modified generators
        network.update_generators(gen_changed[["target_p"]])
        # Calculate the new load flow
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        # Calculate the sensitivity to the parameter
        sensitivity[nom] = (get_flows(network) - initial_flows) / 100
        # Go back to initial generators
        network.update_generators(generators.loc[parameters[nom]["filter"], ["target_p"]])
    elif param["type"] == "loads":
        # Select the loads we want to calculate sensitivity for
        load_changed = loads[parameters[nom]["filter"]].copy()
        # Increase the selected loads by 100 MW (proportionnaly to each load) 
        load_changed["p0"] = load_changed["p0"] + 100 * load_changed["p0"] / load_changed["p0"].sum()
        # Update the network with the modified loads
        network.update_loads(load_changed[["p0"]])
        # Calculate the new load flow
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        # Calculate the sensitivity to the parameter
        sensitivity[nom] = (get_flows(network) - initial_flows) / 100
        # Go back to initial loads
        network.update_loads(loads.loc[parameters[nom]["filter"], ["p0"]])
    elif param["type"] == "exchange":
        # Select the xinjections we want to calculate sensitivity for
        inj_changed = xinjections[parameters[nom]["filter"]].copy()
        # Increase the selected injections by 100 MW (proportionnaly to each xinjections)
        inj_changed["p0"] = inj_changed["p0"] + 100 * inj_changed["p0"] / inj_changed["p0"].sum()
        # Update the network with the modified xinjections
        network.update_dangling_lines(inj_changed[["p0"]])
        # Calculate the new load flow
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        # Calculate the sensitivity to the parameter
        sensitivity[nom] = (get_flows(network) - initial_flows) / 100
        # Go back to initial xinjections values
        network.update_dangling_lines(xinjections.loc[parameters[nom]["filter"], ["p0"]])


# #### PTDF matrix generation

# In[51]:


pd.set_option('display.float_format', '{:.3f}'.format)
PTDF = pd.DataFrame(sensitivity)
PTDF.droplevel(0)


# ## Get yearly TSO values (load, generation, crossborder exchanges)
# 
# Using [Entsoe Transparency Platform at https://transparency.entsoe.eu/](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&dateTime.dateTime=02.09.2024+00:00|CET|DAY&biddingZone.values=CTY|10YFR-RTE------C!CTY|10YFR-RTE------C&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)), it is possible to download the needed data (export to csv files, but using API is also possible):
# ![title](./images/tso_data.png)
# 
# We will download the following data Spain, Portugal and France:
# - yearly **load** (on hourly basis),
# - **generation** per production type,
# - **crossborder exchanges**.

# ### Load data for ES, PT and FR

# In[52]:


TSOS = ["ES", "FR", "PT"]
load = pd.concat(
    [
        pd.read_csv(
            Path("./data") / f"{tso}_Total Load - Day Ahead _ Actual_202101010000-202201010000.csv",
            sep=",",
            index_col=0,
            usecols=lambda x: x.startswith(("Time", "Actual Total Load")),
        )
        for tso in TSOS
    ], axis=1
).reset_index(drop=True)
load = load.rename(columns={col: re.sub(r"[a-zA-Z \[\]-]*\((.*)\)", r'\1_Load', col) for col in load.columns})
load


# In[53]:


fig = load.plot(
    color_discrete_map={col: color for col, color in zip(load.columns, colors)},
    labels=dict(index="Hours", value="Load (MW)", variable="Load")
)
fig.update_layout(
    plot_bgcolor='white'
)
fig.update_xaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.show()


# ### For generation

# In[54]:


generation = pd.concat(
    [
        pd.read_csv(
            Path("./data") / f"{tso}_Actual Generation per Production Type_202101010000-202201010000.csv",
            sep=",",
            index_col=0,
            na_values="n/e"
        )
        .reset_index(drop=True)
        .rename(columns=lambda x: tso + "_" + x)
        for tso in TSOS
    ], axis=1
)
generation.head()


# In[55]:


for tso in TSOS:
    generation[tso + "_Gen_Total"] = generation.filter(regex=tso + ".*Aggregated").sum(axis=1) - generation.filter(regex=tso + ".*Consumption").sum(axis=1)
generation.filter(like="Total")


# In[56]:


generation.head()


# In[57]:


generation = generation.rename(columns=lambda x: x.replace(" - Actual Aggregated [MW]", "").strip())
generation.head()


# In[58]:


col_to_draw = ["ES_Gen_Total", "FR_Gen_Total", "PT_Gen_Total"]
fig = generation[col_to_draw].plot(
    color_discrete_map={col: color for col, color in zip(col_to_draw, colors)},
    labels=dict(index="Hours", value="Generation (MW)", variable="Total Generation")
)
fig.update_layout(
    plot_bgcolor='white'
)
fig.update_xaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.show()


# ### For crossborder exchanges

# In[59]:


links = ["FR_ES", "FR_BE", "FR_CH", "FR_DE", "FR_IT", "FR_UK", "ES_PT"]
crossborders_flows = pd.concat(
    [
        pd.read_csv(Path("./data") / f"{link}_Cross-Border Physical Flow_202101010000-202201010000.csv", sep=",", index_col=0).reset_index(drop=True)
        for link in links
    ], axis=1
)
crossborders_flows = crossborders_flows.rename(columns={col: re.sub(r"[a-zA-Z ]*\((.*)\) > [a-zA-Z ]*\((.*)\) \[MW\]", r'\1_\2', col) for col in crossborders_flows.columns})


# In[60]:


for link in links:
    crossborders_flows[link] = crossborders_flows[link] - crossborders_flows[link.split("_")[1] + "_" + link.split("_")[0]]
crossborders_flows = crossborders_flows[links]
crossborders_flows


# ### Check that balances are correct for each TSO

# #### Calculate the balance using load and generation#

# In[61]:


balance_generation_load = {}
for tso in TSOS:
    balance_generation_load[tso] = generation[f"{tso}_Gen_Total"] - load[f"{tso}_Load"]


# #### Calculate the balance using exchanges

# In[62]:


balance_exchanges = {
    "ES": - crossborders_flows["FR_ES"] + crossborders_flows["ES_PT"],
    "PT": -crossborders_flows["ES_PT"],
    "FR":  (crossborders_flows["FR_ES"] + crossborders_flows["FR_BE"] + crossborders_flows["FR_DE"] + crossborders_flows["FR_CH"]+ crossborders_flows["FR_IT"] + crossborders_flows["FR_UK"])
}


# #### Comparison of balances

# In[63]:


from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=2, shared_xaxes=True, subplot_titles=["ES balances", "ES balances diff", "FR balances", "FR balances diff", "PT balances", "PT balances diff"]) 
for idx, tso in enumerate(TSOS):
    
    fig.add_trace(
        go.Scatter(
            x=balance_exchanges[tso].index,
            y=balance_exchanges[tso].to_numpy(),
            mode='lines',
            marker=dict(color=colors[5]),
            
        ),
        row=1 + idx, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=balance_generation_load[tso].index,
            y=balance_generation_load[tso].to_numpy(),
            mode='lines',
            marker=dict(color=colors[idx]),
        ),
        row=1 + idx, col=1,
    
    )
    fig.add_trace(
        go.Scatter(
            x=balance_generation_load[tso].index,
            y=(balance_generation_load[tso] - balance_exchanges[tso]).to_numpy(),
            mode='lines',
            marker=dict(color=colors[idx]),
            
        ),
        row=1 + idx, col=2,
    )

fig.update_layout(
    plot_bgcolor='white',
    width=1300, height=700,
    showlegend=False,
)
fig.update_xaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    mirror=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.show()


# #### Modify the LOAD in each TSO to have generation - load = 0

# In[64]:


for tso in TSOS:
    load[f"{tso}_Load"] -= (balance_exchanges[tso] - balance_generation_load[tso])
    balance_generation_load[tso] = generation[f"{tso}_Gen_Total"] - load[f"{tso}_Load"]
    print(tso)
    print((balance_exchanges[tso] - balance_generation_load[tso]).mean())


# ### Build TSO data

# In[65]:


generation["FR_Other_Gen"] = generation["FR_Gen_Total"] - generation["FR_Nuclear"]
generation["ES_Other_Gen"] = generation["ES_Gen_Total"] - generation["ES_Wind Onshore"]


# In[66]:


tso_data = pd.concat([load, crossborders_flows, generation["FR_Nuclear"], generation["ES_Wind Onshore"], generation["FR_Other_Gen"], generation["ES_Other_Gen"], generation["PT_Gen_Total"] ], axis=1)
tso_data["ES_MA"] = 0
tso_data["FR_IE"] = 0
tso_data


# ## Calculate yearly flows
# 
# \begin{align}
# flows_{Hour} = flows_{base\_case} + (tso\_data_{Hour} - tso\_data_{base\_case}) . PTDF
# \end{align}

# In[67]:


flows = initial_flows + (tso_data[PTDF.columns] - initial_tso_data).dot(PTDF.T)
flows


# In[68]:


flows["_363fc03a-2307-5959-20f7-38c771355440 + _547b44cb-ecc8-514f-bda9-c73a7671ce38"].plot()


# In[69]:


flows["_0b3075b2-7980-988a-224b-407447636928 + _aaa73dfd-dc95-5acc-bed1-fafc4b3d2d8b"].plot()


# ## Calculate ES-FR transfer capacity in **N condition**
# 

# ### Define lines to monitor
# 
# In this notebook, to simplify the calculations, we will only keep tie-lines between France and Spain
# 
# But to assess transfer capacity in the framework of TYNDP, according to the Implementation Guidelines, all the lines with sensitivity greater than 5% should be taken into account

# In[70]:


tie_lines


# In[71]:


monitored_linenames = [
    "HERNANI_XHE_AR11_1_400 + .HERNL71ARGIA",
    ".ARKAL61ARGIA + XAR_AR21_DESF.ARK_1_220",
    ".BIESL61PRAGN + BIESCAS_XBI_PR21_1_220",
    "VIC_XVI_BA11_1_400 + .VICHL71BAIXA"
]
# Get only FR-ES tie-lines
monitored_lines = tie_lines[tie_lines["name"].isin(monitored_linenames)]
monitored_lines


# ### Calculate sensitivity to new exchanges (= powershift) between ES &rarr; FR

# In[72]:


# Simulate an exchange between ES and FR, based on load variation
powershift_sensitivity = (PTDF['FR_Load'] - PTDF['ES_Load'])
# We will keep only lines with sensitivity greater than 5%
powershift_sensitivity = powershift_sensitivity[powershift_sensitivity.abs() > 0.05]
powershift_sensitivity_with_linenames = powershift_sensitivity.droplevel(0)
powershift_sensitivity_with_linenames.sort_values()


# ### Extract ratings from network model

# #### Get ratings

# In[73]:


ratings = network.get_operational_limits()
ratings


# #### Keep only PATL ratings (permanent admissible transmission loading) 

# In[74]:


patl_ratings = ratings[ratings["acceptable_duration"] == -1]


# #### Extract ratings for monitored lines...

# In[75]:


ratings_monitored_lines = (
    monitored_lines
    # merge with ratings to retrieve value of ratings on both sides of the tie-line
    .merge(patl_ratings["value"], left_on="dangling_line1_id", right_index=True, how="left")
    .merge(patl_ratings["value"], left_on="dangling_line2_id", right_index=True, how="left", suffixes=("1", "2"))
)
ratings_monitored_lines


# ####  ...and convert them in MW

# In[76]:


# Keep lowest value of both sides
ratings_monitored_lines["value"] = ratings_monitored_lines[["value1", "value2"]].fillna(9999).min(axis=1)
ratings_monitored_lines["nominal_v"] = ratings_monitored_lines["nominal_v1"].where(ratings_monitored_lines["value2"].isnull(), ratings_monitored_lines["nominal_v2"])
# Convert the rating from Ampers to MW
ratings_monitored_lines["value_in_mw"] = math.sqrt(3) * ratings_monitored_lines["nominal_v"] * ratings_monitored_lines["value"] / 1_000
ratings_dict = ratings_monitored_lines.set_index("name")["value_in_mw"].to_dict()


# In[77]:


pd.Series(ratings_dict)


# ### Optimization problem
# 
# In N situation, the constraints consist only of flows that shouldn't exceed the N ratings
# 
# \begin{align}
#     - ratings(i) < \overbrace {flow(i, Hour) + powershift(Hour).sensitivity(i)}^\text{flow of line i after powershift} < ratings(i)
# \end{align}
# 
# or
# 
# \begin{align}
#     - ratings(i) - flow(i, Hour) < powershift(Hour).sensitivity(i) < ratings(i) - flow(i, Hour)
# \end{align}
# 
# #### Objective (cost) function
# 
# - maximizing the $powershift$ to calculate NTC A->B
# 
# #### Choice of the solver
# 
# In order to solve the problem, we will use an Open Source Solver Suite: **Google OR-Tools**.
# 
# From OR-Tools, we will use the GLOP solver.This solver has a few advantages:
# - free (no licence needed),
# - easy to install (comes with OR-Tools),
# - building the problem is really easy to implement and easy to understand,
# - seems relatively quick.
# 
# You can find more information on this solver and step by step examples on how to build an optimization problem [here](https://developers.google.com/optimization/introduction/python)

# ### Calculate maximum powershift in N condition

# In[78]:


from ortools.linear_solver import pywraplp

N_RATING_SECURITY_MARGIN = 0.9

# Get the flows and remove id for clarity
flows_n = flows.droplevel(0, axis=1)

ratings = N_RATING_SECURITY_MARGIN * pd.DataFrame(ratings_dict, index=flows_n.index)

solutions = []
# Calculate lower and upper bound to feed the optimizer
lower_bound = pd.DataFrame(0.0, columns=monitored_linenames, index=flows_n.index)
upper_bound = pd.DataFrame(0.0, columns=monitored_linenames, index=flows_n.index)
for monitored_line in tqdm(monitored_linenames, desc="calculate upper/lower bounds"):
    lower_bound[monitored_line] = -ratings[monitored_line] - flows_n[monitored_line]
    upper_bound[monitored_line] = ratings[monitored_line] - flows_n[monitored_line]
    
# For each PiT, calculate the highest (lowest) powershift that respect all the N constraints
for pit in tqdm(range(len(flows)), desc="powershift calculation"):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # Variable used to optimize the cost function: only powershift here
    powershift = solver.NumVar(-solver.infinity(), solver.infinity(), 'powershift')

    # Build the constraints
    constraints = {}
    for iconstraint, monitored_line in enumerate(monitored_linenames):
        # Define each constraint: first, set lower bounds and upper bounds
        constraints[iconstraint] = solver.Constraint(float(lower_bound.loc[pit, monitored_line]), float(upper_bound.loc[pit, monitored_line]))
        # Set how the constraint changes regarding the powershift
        constraints[iconstraint].SetCoefficient(powershift, float(powershift_sensitivity_with_linenames[monitored_line]))
    # Find the maximum Powershift A->B    
    solver.Maximize(powershift)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        solution_max = powershift.solution_value()
    else:
        solution_max = math.nan
    solutions.append(solution_max)


# In[79]:


max_powershift_n = pd.DataFrame(solutions, columns=["max powershift ES→FR in N"])
max_powershift_n.plot(labels=dict(index="Hours", value="MW"))


# ## Calculate ES-FR transfer capacity in **N-1 condition**

# ### Calculate LODF matrix to take into account contingencies

# #### LODF introduction
# 
# Line Outage Distribution Factors (LODFs) represent the percentage of a line flow that will show up in other lines after the outage of this line.
# 
# For example, in case of line_x outage (with initial flow of 100 MW), the following LODF would mean:
# - line_y: **10%** &rarr; **+ 10 MW** on line_y
# - line_z: **- 30%** &rarr; **- 30 MW** on line_y

# #### Formula used to calculate new flow for line i after outage of line j will be:
# 
# 
# \begin{align}
#     flow(i / j, \small{Hour}) = flow(i, \small{Hour}) + LODF(i,j) . flow(j, \small{Hour})
# \end{align}

# #### DC Sensitivity analysis initialization in pyPowsybl
# - define **monitored lines**: FR-ES **tie-lines**
# - define **contingencies**: lines with **sensitivity greater than 5%** to ES-FR powershift
# 
# For a more detailed description, you can have a look at the [sensitivity analysis documentation](https://pypowsybl.readthedocs.io/en/stable/user_guide/sensitivity.html).

# In[80]:


monitored_lines


# In[81]:


contingencies_id = powershift_sensitivity.droplevel(1).index


# In[82]:


# DC analysis initialization
sa = pp.sensitivity.create_dc_analysis()
sa.add_branch_flow_factor_matrix(
    list(monitored_lines.index), [loads.index[0]], "lodf"
)
for contingency in contingencies_id:
    sa.add_single_element_contingency(contingency)


# #### Run the DC sensitivity analysis
# 
# Then get N-1 flows for all the contingencies, and concatenate the results

# In[83]:


sa_result = sa.run(network, parameters_lf)
n_1 = pd.concat(
    [
        sa_result.get_reference_matrix("lodf", contingency)
        for contingency in contingencies_id
    ],
)
n_1.index = contingencies_id


# In[84]:


n_1


# #### Calculate the LODF

# 
# \begin{align}
# LODF^{cb}_{co} = \frac{flow^{cb}_{co} - flow^{cb}_{0}}{flow^{co}_{0}}
# \end{align}
# 
# where:
# - $flow^{cb}_{co}$ is the flow of the critical branch **cb** after the critical outage **co** occurs
# - $flow^{cb}_{0}$ is the initial flow of the critical branch **cb**
# - $flow^{co}_{0}$ is the initial flow of the critical branch **co**

# In[85]:


n_1.sub(sa_result.get_reference_matrix("lodf").squeeze(), axis=1)


# In[86]:


initial_flows.loc[contingencies_id].droplevel(0)


# In[87]:


lodf_matrix = (
    n_1.sub(sa_result.get_reference_matrix("lodf").squeeze(), axis=1).rename(index=lines_id_dict).rename(columns=lines_id_dict)
    .divide(initial_flows.loc[contingencies_id].droplevel(0), axis=0)
    .fillna(0)
)
lodf_matrix


# In[88]:


contingencies_names = list(powershift_sensitivity_with_linenames.index)


# In[89]:


monitored_lines


# ## Optimization problem
# 
# \begin{align}
#     - ratings(i) < \underbrace{\overbrace{flow(i, \small Hour) + powershift(\small Hour).sensitivity(i)}^\text{flow of line i after powershift} + LODF(i,j) . \overbrace{\bigl(flow(j, \small Hour) + powershift(\small Hour).sensitivity(j)\bigr)}^\text{flow of line j after powershift}}_\text{flow of line i after contingency of line j and powershift} < ratings(i)
# \end{align}
# 
# or
# 
# \begin{align}
#     - ratings(i) - flow(i, \small Hour) -  LODF(i, j) . flow(j, \small Hour) < powershift(\small Hour).\bigl(sensitivity(i) + LODF(i,j).sensitivity(j)\bigr) < ratings(i) - flow(i, \small Hour) -  LODF(i, j) . flow(j, \small Hour)
# \end{align}

# ## Calculate maximum powershift in N-1

# In[90]:


N_RATING_SECURITY_MARGIN = 0.9
n_ratings = N_RATING_SECURITY_MARGIN * pd.DataFrame(ratings_dict, index=flows_n.index)
n_1_ratings = pd.DataFrame(ratings_dict, index=flows_n.index)

cbcos_idx = pd.MultiIndex.from_product(
    [list(monitored_linenames), contingencies_names]
)
upper_bounds = pd.DataFrame(0.0, columns=cbcos_idx, index=flows_n.index)
lower_bounds = pd.DataFrame(0.0, columns=cbcos_idx, index=flows_n.index)

otdf_matrix = pd.Series(0.0, index=cbcos_idx)
print("Preparing the optimization problem...")
for monitored_line in tqdm(monitored_linenames):
    for contingency in contingencies_names:
        cbco = (monitored_line, contingency)
        otdf = (
            powershift_sensitivity_with_linenames.loc[monitored_line]
            + lodf_matrix.loc[contingency, monitored_line]
            * powershift_sensitivity_with_linenames.loc[contingency]
        )
        otdf_matrix[(monitored_line, contingency)] = otdf
        if monitored_line == contingency:
            # N condition
            upper_bounds[(monitored_line, contingency)] = (
                n_ratings[monitored_line] - flows_n[monitored_line]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_ratings[monitored_line] - flows_n[monitored_line]
            )
        else:
            # N-1 condition
            upper_bounds[(monitored_line, contingency)] = (
                n_1_ratings[monitored_line]
                - flows_n[monitored_line]
                - lodf_matrix.loc[contingency, monitored_line] * flows_n[contingency]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_1_ratings[monitored_line]
                - flows_n[monitored_line]
                - lodf_matrix.loc[contingency, monitored_line] * flows_n[contingency]
            )

powershift_result = []
critical_branches = []
print("Computing NTC...")
for pit in tqdm(flows.index):
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Variable used to optimize the cost function: only powershift here
    powershift = solver.NumVar(-solver.infinity(), solver.infinity(), "powershift")    
    constraints = []
    for iconstraint, cbco in enumerate(cbcos_idx):
        monitored_line, contingency = cbco
        # Define each constraint: first, set lower bounds and upper bounds
        constraints.append(
            solver.Constraint(
                float(lower_bounds.at[pit, cbco]),
                float(upper_bounds.at[pit, cbco]),
                monitored_line
                + "/"
                + contingency,
            )
        )
        # Set how the constraint changes regarding the powershift
        constraints[iconstraint].SetCoefficient(
            powershift, float(otdf_matrix[iconstraint])
        )

    # Max powershift in A->B direction
    solver.Maximize(powershift)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        powershift_sol = powershift.solution_value()
        # Extract the critical branch that is limiting the powershift max for this PiT
        critical_branch = "+".join([
            constraint.name()
            for constraint in constraints
            if abs(constraint.dual_value()) > 0
        ])
    else:
        critical_branch = "no solution"
        powershift_sol = math.nan

    powershift_result.append(powershift_sol)
    critical_branches.append(critical_branch)


# In[91]:


pd.DataFrame(powershift_result).head()


# In[92]:


powershift_n_1 = pd.DataFrame(powershift_result, columns=["powershift_max (N-1)"])
powershift_n_1.plot(labels=dict(index="Hours", value="MW"))


# ## Assess HVDC impact on transfer capacity

# ### Create HVDC Baixas-SLlogaia

# In[93]:


display_voltage_level("LLOGAIA")


# In[102]:


voltage_level_from_id = voltage_levels[voltage_levels["name"] == "BAIXAP7"].index[0]
voltage_level_to_id = voltage_levels[voltage_levels["name"] == "LLOGAIA"].index[0]
pp.network.create_vsc_converter_station_bay(
    network,
    id=["Baixas" + '_VSC1', "SLlogaia" + '_VSC2'],
    voltage_regulator_on=[True, True],
    loss_factor=[0, 0],
    target_v=[400, 400],
    bus_or_busbar_section_id=[network.get_bus_breaker_topology(voltage_level_from_id).buses.index[0], network.get_bus_breaker_topology(voltage_level_to_id).buses.index[0]],
    position_order=[1000, 1000],
    raise_exception=True,
)
network.create_hvdc_lines(
    id="BaixasSLlogaia",
    converter_station1_id="Baixas" + "_VSC1",
    converter_station2_id="SLlogaia" + "_VSC2",
    r=0,
    nominal_v=320,
    converters_mode='SIDE_1_RECTIFIER_SIDE_2_INVERTER',
    max_p=2000,
    target_p=0,
)


# In[94]:


display_voltage_level("LLOGAIA")


# In[95]:


display_voltage_level("BAIXAP7")


# ### Calculate sensitivity of lines to HVDC
# 
# We want to know what is the impact on AC lines of increasing the HVDC setpoint by 1 MW.
# 
# We will use the ``dc_analysis`` functionality provided by pyPowSyBl.  

# In[96]:


sa = pp.sensitivity.create_dc_analysis()
monitored_elements = sorted(set([
    line
    for line in list(contingencies_id)
    if line not in xinjections.index
]))
sa.add_branch_flow_factor_matrix(monitored_elements, ["BaixasSLlogaia"], "hvdc")
sa_result = sa.run(network, parameters_lf)
hvdc_sensitivity = sa_result.get_sensitivity_matrix("hvdc").T.rename(index=lines_id_dict)


# In[97]:


hvdc_sensitivity.sort_values(by='BaixasSLlogaia')


# ### Calculate LODF for HVDC

# We have already calculated the HVDC sensitivity which is the impact on AC lines of increasing the HVDC setpoint by 1 MW (HVDC from 0 MW to 1 MW for example).
# 
# The impact of an HVDC outage on the grid will be the opposite: a N-1 HVDC with initial setpoint at 1 MW can also be seen as an HVDC setpoint going from 1 MW to 0 MW.
# 
# &rarr;**LODF for HVDC will be the opposite of its sensitivity**

# In[98]:


n_1_hvdc = - hvdc_sensitivity
n_1_hvdc = n_1_hvdc.loc[monitored_linenames]
n_1_hvdc.T


# ### Add LODF of HVDC to the global LODF matrix

# In[99]:


lodf_matrix = pd.concat([lodf_matrix, n_1_hvdc.T])
lodf_matrix


# ### Calculate ES-FR transfer capacity in **N-1 condition**, with HVDC optimization

# - Add the HVDC to the liste of contingencies

# In[100]:


contingencies_with_hvdc = contingencies_names + ["BaixasSLlogaia"]


# - HVDC are not sensitive to powershift

# In[101]:


powershift_sensitivity_with_linenames.loc["BaixasSLlogaia"] = 0


# - Initial setpoint of HVDC is 0 MW

# In[102]:


flows_n["BaixasSLlogaia"] = 0


# ### Build the optimization problem and solve it
# 
# In order to optimize the NTC allowed by the grid, it can be interesting to take advantage of HVDC installed near the border taken into account.
# 
# To calculate the impact of HVDC on the flows on every line of the network, we can use the sensitivites calculated for this kind of device.

# ##### Constraints
# The constraints in N/N-1 condition with HVDC optimization will consist of:
# - flows including impact of powershift and HVDC changes shouldn't exceed ratings of lines
# - Capacity of HVDC

# ##### Constraints on flows
# 
# \begin{align}
#     - ratings(i) < \underbrace { \overbrace { flow(i,\small Hour) + \alpha_i . powershift(\small Hour) + \beta_i . HVDC({\small Hour})}^\text{flow of line i after powershift and HVDC changes} + LODF(i,j) . \bigl(\overbrace { flow(j,{\small Hour}) + \alpha_j . powershift({\small Hour}) +  \beta_j . HVDC({\small Hour})}^\text{flow of line j after powershift and HVDC changes}\bigr)}_\text{flow of line i after contingency of line j, powershift and HVDC changes} < ratings(i)
# \end{align}
# 
# where:
# - $\alpha_i$ is the sensibility of the $line_i$ towards variation of the powershift
# - $\beta_i$ is the sensibility of the $line_i$ towards variation of the
#       HVDC: variation of 1 MW of set-point for HVDC.
# - $HVDC$ = new value for the target set-point value for HDVC, calulated to optimize the Powershift
#  
# or
# 
# \begin{align}
#     - ratings(i) - flow(i, \small Hour) -  LODF(i, j) . flow(j, \small Hour) < powershift(\small Hour).\overbrace { \bigl( \alpha_i + LODF(i,j).\alpha_j \bigr)}^\text{OTDF(i, j)} + HVDC(\small Hour). \overbrace { \bigl(\beta_i + LODF(i,j).\beta_j\bigr)}^\text{OTDF\_HVDC(i, j)} < ratings(i) - flow(i, \small Hour) -  LODF(i, j) . flow(j, \small Hour)
# \end{align}
# 

# In[103]:


N_RATING_SECURITY_MARGIN = 0.9

n_ratings = N_RATING_SECURITY_MARGIN * pd.DataFrame(ratings_dict, index=flows_n.index)
n_1_ratings = pd.DataFrame(ratings_dict, index=flows_n.index)

cbcos_idx = pd.MultiIndex.from_product(
    [list(monitored_linenames), contingencies_with_hvdc]
)
otdf_hvdc = pd.DataFrame([], columns=cbcos_idx, index=["BaixasSLlogaia"])

upper_bounds = pd.DataFrame(0.0, columns=cbcos_idx, index=flows_n.index)
lower_bounds = pd.DataFrame(0.0, columns=cbcos_idx, index=flows_n.index)

otdf_matrix = pd.Series(0.0, index=cbcos_idx)
print("Preparing the optimization problem...")
for monitored_line in tqdm(monitored_linenames):
    for contingency in contingencies_with_hvdc:
        cbco = (monitored_line, contingency)
        if contingency == "BaixasSLlogaia":
            # in case of N-1 HVDC, the HVDC has no more influence to the lines
            otdf_hvdc.loc["BaixasSLlogaia", cbco] = 0
        else:
            otdf_hvdc.loc["BaixasSLlogaia", cbco] = (
                hvdc_sensitivity.loc[monitored_line, "BaixasSLlogaia"]
                + lodf_matrix.loc[contingency, monitored_line]
                * hvdc_sensitivity.loc[contingency, "BaixasSLlogaia"]
            )

        otdf = (
            powershift_sensitivity_with_linenames.loc[monitored_line]
            + lodf_matrix.loc[contingency, monitored_line]
            * powershift_sensitivity_with_linenames.loc[contingency]
        )
        otdf_matrix[(monitored_line, contingency)] = otdf
        if monitored_line == contingency:
            # N condition
            upper_bounds[(monitored_line, contingency)] = (
                n_ratings[monitored_line] - flows_n[monitored_line]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_ratings[monitored_line] - flows_n[monitored_line]
            )
        else:
            # N-1 condition
            upper_bounds[(monitored_line, contingency)] = (
                n_1_ratings[monitored_line]
                - flows_n[monitored_line]
                - lodf_matrix.loc[contingency, monitored_line] * flows_n[contingency]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_1_ratings[monitored_line]
                - flows_n[monitored_line]
                - lodf_matrix.loc[contingency, monitored_line] * flows_n[contingency]
            )

powershift_result_with_hvdc = []
hvdc_result = []
critical_branches_with_hvdc = []
print("Computing NTC...")
for pit in tqdm(flows.index):
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Variable used to optimize the cost function: powershift here
    powershift = solver.NumVar(-solver.infinity(), solver.infinity(), "powershift")
    # And also the HVDC
    hvdc = solver.NumVar(-2000, 2000, "BaixasSLlogaia")
    constraints = []
    for iconstraint, cbco in enumerate(cbcos_idx):
        monitored_line, contingency = cbco
        # Define each constraint: first, set lower bounds and upper bounds
        constraints.append(
            solver.Constraint(
                float(lower_bounds.at[pit, cbco]),
                float(upper_bounds.at[pit, cbco]),
                monitored_line
                + "/"
                + contingency,
            )
        )
        # Set how the constraint changes regarding the powershift
        constraints[iconstraint].SetCoefficient(
            powershift, float(otdf_matrix[iconstraint])
        )
        constraints[iconstraint].SetCoefficient(hvdc, float(otdf_hvdc.at["BaixasSLlogaia", cbco]))

    # Max powershift in A->B direction
    solver.Maximize(powershift)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        powershift_sol = powershift.solution_value()
        hvdc_sol = hvdc.solution_value()
        # Extract the critical branch that is limiting the powershift max for this PiT
        critical_branch = "+".join([
            constraint.name()
            for constraint in constraints
            if abs(constraint.dual_value()) > 0
        ])
    else:
        critical_branch = "no solution"
        powershift_sol = math.nan
        hvdc_sol = math.nan

    powershift_result_with_hvdc.append(powershift_sol)
    hvdc_result.append(hvdc_sol)

    critical_branches_with_hvdc.append(critical_branch)
    


# In[104]:


powershift_n_1_with_hvdc = pd.DataFrame(powershift_result_with_hvdc, columns=["max_powershift (N-1 + HVDC)"])
powershift_n_1_with_hvdc.plot(labels=dict(index="Hours", value="MW"))


# ## Draw the ΔNTC curve

# In[105]:


import numpy as np
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 1, len(powershift_n_1_with_hvdc["max_powershift (N-1 + HVDC)"])),
        y=pd.DataFrame((powershift_n_1_with_hvdc["max_powershift (N-1 + HVDC)"] - powershift_n_1["powershift_max (N-1)"]).sort_values()).squeeze(),
        name="ΔNTC ES→FR",
        mode="lines",
        hovertemplate="%{x:.2f} %<br>ΔNTC: %{y:.0f} MW<br>",
        showlegend=True,
    )
)
fig.update_xaxes(
    tickformat=".1%",
    ticks="outside",
    title={"standoff": 15, "text": "Time (%)"},
)


# <div class="alert alert-block alert-info">
# The steps presented in this notebook are relatively close to what is done to perform &Delta;NTC calculations. However, for the sake of simplicity, a few approximations have been made:
# 
# - only tie-lines are monitored in this notebook: normally all lines with sensitivity greater than 5% should be monitored,
# - country data (load/generation/exchanges) are taken for year 2021: not in line with the TYNDP2022 models (built for 2030 time horizon)
# - in TYNDP network studies, we don't rely on historical data, but rather perform market simulations based on Entsoe scenarios: Antares (https://github.com/AntaresSimulatorTeam/Antares_Simulator), also OpenSource, is one of these tools,
# - only nuclear units in France / wind generation in Spain have been fully modelized: we usually modelize each generation per production type,
# - PST should also be optimized (the same way as HVDC) when they are close to the new interconnection project we assess,
# - N-1 ratings should be taken into account for internal lines and for some TSO, seasonal thermal limits.
# </div>
