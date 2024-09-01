#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import math
import re
from pathlib import Path

from tqdm.notebook import tqdm
from IPython.display import SVG
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import pypowsybl as pp
pd.options.plotting.backend = "plotly"

try:
    import entsoe_secrets
    token = entsoe_secrets.token
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


# ## Quick tour on how to use pyPowSybl
# All the elements of the network can be accessed by using the following methods :
# - voltage levels: ``network.get_voltage_levels()`` (&rightarrow; columns **name**, **substation_id**, **nominal_v**),
# - substations: ``network.get_substations()`` (&rightarrow; columns **name**, **country**),
# - loads: ``network.get_loads()`` (&rightarrow; columns **p0**, **bus_id**, **voltage_level_id**)
# - generators: ``network.get_generators()``
# - pst: ``network.get_phase_tap_changers()``,
# - buses: ``network.get_buses()`` (&rightarrow; columns **name**, **synchonous_component**)
# - lines: ``network.get_lines()`` (&rightarrow; columns **name**, **voltage_level1_id**, **voltage_level2_id**, **connected1**, **connected2**),
# - two windings transformers: ``network.get_2_windings_transformers()`` (&rightarrow; columns **name**, **voltage_level1_id**, **voltage_level2_id**, **connected1**, **connected2**),

# ### Load a test case MicroGrid

# In[2]:


network_test = pp.network.create_micro_grid_be_network()


# In[3]:


network_test.get_network_area_diagram([], nad_parameters=pp.network.NadParameters(edge_name_displayed=False, substation_description_displayed=True, edge_info_along_edge=True))


# ### Substations
# 
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


# # Load the French, Spanish and Portuguese networks using pyPowsyBl

# In[13]:


boundary_set = "20211216T1459Z_ENTSO-E_BD_1346.zip"
parameters = {
    "iidm.import.cgmes.boundary-location": str((Path("./data") / boundary_set).resolve()),    # the boundary set will be imported
    "iidm.import.cgmes.source-for-iidm-id": "rdfID",        # rdfID will be used as id for elements
}
models = {
    "ES": "ES_3PQT.zip",
    "FR": "FR_3PQT_v3.zip",
    "PT": "PT_3PQT_v2.zip",
}
networks = {}
for tso, model in tqdm(models.items()):
    networks[tso] = pp.network.load(Path("./data") / model, parameters=parameters)
network = networks["FR"]
network.merge([networks["ES"], networks["PT"]])


# ## Build interesting dataframes

# ### For voltage_levels
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

# In[14]:


voltage_levels = network.get_voltage_levels(attributes=["name", "substation_id", "nominal_v"])
voltage_levels.head()


# In[15]:


substations = network.get_substations(attributes=["name", "CGMES.regionName", "geo_tags"])
substations.head()


# In[16]:


voltage_levels = voltage_levels.merge(substations, left_on="substation_id", right_index=True, suffixes=("", "_subst"))
voltage_levels.head()


# In[17]:


buses = network.get_buses(attributes=["synchronous_component"])


# ### For generators
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

# In[18]:


generators = network.get_generators(attributes=["name", "energy_source", "target_p", "max_p", "bus_id", "voltage_level_id"]) 
generators = generators.merge(buses, left_on="bus_id", right_index=True, how="left")
generators = generators.merge(voltage_levels, left_on="voltage_level_id", right_index=True, suffixes=("", "_vl"))
generators = generators[generators["synchronous_component"] == 0]
generators.head()


# ### For loads
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

# In[19]:


loads = network.get_loads(attributes=["p0", "bus_id", "voltage_level_id"])
loads = loads.merge(buses, left_on="bus_id", right_index=True, how="left")
loads = loads.merge(voltage_levels, left_on="voltage_level_id", right_index=True, suffixes=("", "_vl"))
loads = loads[loads["synchronous_component"] == 0]
loads.head()


# ### Dangling lines / External injections

# In[20]:


xinjections = network.get_dangling_lines(attributes=["name", "p0", "p", "voltage_level_id", "tie_line_id"])
xinjections[xinjections["name"].isin([".EICL72MUHL_PINT228", ".ENSDL71VIGY", ".ENSDL72VIGY", ".EICL73MUHL_PINT228", ])]


# In[21]:


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


# In[22]:


display_voltage_level("VIGY P7")


# ## Geographical representation of the network near the France - Spain border

# ### Using OpenData (https://odre.opendatasoft.com/explore) to get RTE substations coordinatees

# In[23]:


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


# All RTE substations coordinates can be found at https://odre.opendatasoft.com/explore/dataset/postes-electriques-rte/table/?disjunctive.fonction&disjunctive.etat&disjunctive.tension&sort=-code_poste

# In[24]:


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


# ### Using geocoder service to find GPS coordinates of Red Electrica substations

# In[25]:


idx_es = network.get_network_area_diagram_displayed_voltage_levels(
    # Only substations close to the border
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


# In[26]:


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


# In[27]:


gps_coords_es = {}
to_execute = False
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


# In[28]:


coords_es = subst_for_gps_es.merge(pd.DataFrame.from_dict(gps_coords_es, orient="index", columns=["latitude", "longitude"]), left_on="name", right_index=True)
coords_es.head()


# In[29]:


coords_fr = gps_coords_fr.merge(subst_for_gps_fr.reset_index(), left_on="Code poste", right_on="name").set_index("substation_id")
coords_fr.head()


# In[30]:


coords_fr_es = pd.concat([coords_fr, coords_es]).rename_axis("id")[["latitude", "longitude"]]
coords_fr_es.head()


# In[31]:


network.remove_extensions('substationPosition', network.get_extensions('substationPosition').index)
network.create_extensions('substationPosition', coords_fr_es[["latitude", "longitude"]])
lines = network.get_lines(["connected1", "connected2"])
network.remove_elements(lines[~lines["connected1"] | ~lines["connected2"]].index)
try:
    import ipywidgets as widgets
    from pypowsybl_jupyter import NetworkMapWidget, display_sld
    out_events = widgets.Output()
    mapview = NetworkMapWidget(network, nominal_voltages_top_tiers_filter=5, use_name=True)
except ImportError:
    parameter = pp.network.NadParameters(layout_type=pp.network.NadLayoutType.GEOGRAPHICAL, bus_legend=False)
    mapview = SVG(network.get_network_area_diagram(list(vl_for_gps_fr.index.union(vl_for_gps_es.index)), nad_parameters=parameter).svg.replace("25px", "250px").replace("stroke-width: 5", "stroke-width: 15"))
mapview


# # Calculate yearly flows in the grid

# ## Network modification

# ### Create the slack bus

# In[32]:


# Create substation
station = pd.DataFrame.from_records(
    index="id",
    data=[
        {"id": "SLACK_SUBST", "country": "TN"},
    ],
)
network.create_substations(station)

# Create voltage level
voltage_level = pd.DataFrame.from_records(
    index="id",
    data=[
        {
            "substation_id": "SLACK_SUBST",
            "id": "SLACK_VL",
            "topology_kind": "BUS_BREAKER",
            "nominal_v": 380,
        },
    ],
)
network.create_voltage_levels(voltage_level)

# Create node
network.create_buses(id=['SLACK_NODE'], voltage_level_id=['SLACK_VL'])

# Create load
load = pd.DataFrame.from_records(
    index="id",
    data=[
        {
            "voltage_level_id": "SLACK_VL",
            "id": "SLACK_LOAD",
            "bus_id": "SLACK_NODE",
            "p0": 5000,
            "q0": 0,
        }
    ],
)
network.create_loads(load)

# Create slack generator
generator = pd.DataFrame.from_records(
    index="id",
    data=[
        {
            "voltage_level_id": "SLACK_VL",
            "id": "SLACK_GROUPE",
            "bus_id": "SLACK_NODE",
            "target_p": 5000,
            "min_p": -100000,
            "max_p": 100000,
            "target_v": 380,
            "voltage_regulator_on": True,
        }
    ],
)
network.create_generators(generator)

# Create the SLACK LINE, connected to TRIP.P7 voltage_level
voltage_level_connection = "TRI.PP7"
voltage_level_connection_id = voltage_levels[voltage_levels["name"] == voltage_level_connection].index[0]
network.create_lines(
    id='SLACK_LINE',
    voltage_level1_id='SLACK_VL',
    bus1_id='SLACK_NODE',
    voltage_level2_id=voltage_level_connection_id,
    bus2_id=network.get_bus_breaker_topology(voltage_level_connection_id).buses.index[0],
    b1=0, b2=0, g1=0, g2=0, r=0.5, x=10
)


# In[33]:


parameters_lf = pp.loadflow.Parameters(
    distributed_slack=False,
    provider_parameters={
        "slackBusSelectionMode": "NAME",
        "slackBusesIds": "SLACK_VL",
        "plausibleActivePowerLimit": "20000"
    },
    connected_component_mode=pp.loadflow.ConnectedComponentMode.MAIN,
)
pp.loadflow.run_dc(network, parameters=parameters_lf)


# In[34]:


# parameters_lf = pp.loadflow.Parameters(
#     distributed_slack=True,
#     connected_component_mode=pp.loadflow.ConnectedComponentMode.MAIN,
# )


# In[35]:


display_voltage_level("SLACK_VL")


# ### Create HVDC Baixas-SLlogaia

# In[36]:


# voltage_level_from_id = voltage_levels[voltage_levels["name"] == "BAIXAP7"].index[0]
# voltage_level_to_id = voltage_levels[voltage_levels["name"] == "LLOGAIA"].index[0]
# pp.network.create_vsc_converter_station_bay(
#     network,
#     id=["Baixas" + '_VSC1', "SLlogaia" + '_VSC2'],
#     voltage_regulator_on=[True, True],
#     loss_factor=[0, 0],
#     target_v=[400, 400],
#     bus_or_busbar_section_id=[network.get_bus_breaker_topology(voltage_level_from_id).buses.index[0], network.get_bus_breaker_topology(voltage_level_to_id).buses.index[0]],
#     position_order=[1000, 1000],
#     raise_exception=True,
# )
# network.create_hvdc_lines(
#     id="BaixasSLlogaia",
#     converter_station1_id="Baixas" + "_VSC1",
#     converter_station2_id="SLlogaia" + "_VSC2",
#     r=0,
#     nominal_v=320,
#     converters_mode='SIDE_1_RECTIFIER_SIDE_2_INVERTER',
#     max_p=2000,
#     target_p=0,
# )


# In[37]:


# network.create_hvdc_lines(
#     id="BaixasSLogaia",
#     converter_station1_id="BaixasSLogaia" + "_VSC1",
#     converter_station2_id="BaixasSLogaia" + "_VSC2",
#     r=0,
#     nominal_v=320,
#     converters_mode='SIDE_1_RECTIFIER_SIDE_2_INVERTER',
#     max_p=2000,
#     target_p=0,
# )


# In[38]:


# display_voltage_level("LLOGAIA")


# ### Remove generators modelizing HVDC injections

# In[39]:


#hvdc = generators["name"].str.contains("VSC") | generators["name"].str.contains("HVDC")
hvdc = generators["name"].str.contains("VSC")
generators[hvdc]


# In[40]:


network.remove_elements(generators[hvdc].index)
generators = generators.loc[~hvdc]


# In[41]:


# parameters_lf = pp.loadflow.Parameters(
#     distributed_slack=False,
#     connected_component_mode=pp.loadflow.ConnectedComponentMode.MAIN,
# )
# res = pp.loadflow.run_dc(network, parameters=parameters_lf)
# res


# ## Calculate sensitivity of lines to parameters : PTDF matrix

# #### Example with load increase in Spain

# ##### Calculate initial flows

# In[42]:


def get_flows(network):
    lines = network.get_lines(attributes=["name", "p1"])
    tie_lines = network.get_dangling_lines(attributes=["name", "p"]).rename(columns={"p": "p1"})
    flows = pd.concat([lines, tie_lines])
    return flows.reset_index().set_index(["id", "name"])["p1"]


# In[43]:


res = pp.loadflow.run_dc(network, parameters=parameters_lf)
initial_flows = get_flows(network)
res


# ##### Increase the load in Spain by 100 MW

# In[44]:


load_es = loads[loads["CGMES.regionName"] == "ES"].copy()
load_es["p0"] = load_es["p0"] + 100 * load_es["p0"] / load_es["p0"].sum()
network.update_loads(load_es[["p0"]])


# ##### Perform new load flow after load increase

# In[45]:


res = pp.loadflow.run_dc(network, parameters=parameters_lf)
flows_after_load_increase = get_flows(network)
res


# ##### Calculate sensitivity to load increase in Spain

# In[46]:


sensitivity = (flows_after_load_increase - initial_flows) / 100
sensitivity.dropna().sort_values()


# ##### Go back to initial values

# In[47]:


network.update_loads(loads.loc[loads["CGMES.regionName"] == "ES", ["p0"]])
res = pp.loadflow.run_dc(network, parameters=parameters_lf)
res


# ### Parameters selection

# In[48]:


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


# ### Initial values of the selected parameters

# In[49]:


initial_tso_data = {}
balance = 0
for nom, param in parameters.items():
    if param["type"] == "generators":
        initial_tso_data[nom] = generators.loc[param["filter"], "target_p"].sum()
        balance += initial_tso_data[nom]
    elif param["type"] == "loads":
        initial_tso_data[nom] = loads.loc[param["filter"], "p0"].sum()
        balance -= initial_tso_data[nom]
    elif param["type"] == "exchange":
        initial_tso_data[nom] = xinjections.loc[param["filter"], "p0"].sum()
        balance -= initial_tso_data[nom]
initial_tso_data = pd.Series(initial_tso_data)


# ### Check that balance is correct

# In[50]:


balance = (
    initial_tso_data[["ES_Wind Onshore", "ES_Other_Gen", "FR_Nuclear", "FR_Other_Gen", "PT_Gen_Total"]].sum()
    - initial_tso_data[["FR_BE", "FR_IT", "FR_UK", "FR_DE", "FR_CH", "ES_MA", "FR_IE"]].sum()
    - initial_tso_data[["FR_Load", "ES_Load", "PT_Load"]].sum()
)
balance


# ##### Increase the load in Spain by 100 MW

# In[51]:


# increase load in Spain
load_es = loads[parameters["ES_Load"]["filter"]].copy()
load_es["p0"] = load_es["p0"] + 100 * load_es["p0"] / load_es["p0"].sum()
network.update_loads(load_es[["p0"]])


# ##### New load flow

# In[52]:


res = pp.loadflow.run_dc(network, parameters=parameters_lf)
print(res[0].slack_bus_results[0].active_power_mismatch)
new_flows = get_flows(network)


# ##### Go back to initial loads

# In[53]:


network.update_loads(loads.loc[parameters["ES_Load"]["filter"], ["p0"]])


# ##### Calculate sensitivity to load increase in Spain

# In[54]:


(new_flows - initial_flows).dropna().sort_values()


# ### Calculating sensitivities to all selected parameters

# In[55]:


flows = {}
for nom, param in parameters.items():
    if param["type"] == "generators":
        gen_changed = generators[parameters[nom]["filter"]].copy()
        gen_changed["target_p"] = gen_changed["target_p"] + 100 * gen_changed["max_p"] / gen_changed["max_p"].sum()
        network.update_generators(gen_changed[["target_p"]])
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        flows[nom] = get_flows(network) - initial_flows
        network.update_generators(generators.loc[parameters[nom]["filter"], ["target_p"]])
    elif param["type"] == "loads":
        load_changed = loads[parameters[nom]["filter"]].copy()
        load_changed["p0"] = load_changed["p0"] + 100 * load_changed["p0"] / load_changed["p0"].sum()
        network.update_loads(load_changed[["p0"]])
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        flows[nom] = get_flows(network) - initial_flows
        network.update_loads(loads.loc[parameters[nom]["filter"], ["p0"]])
    elif param["type"] == "exchange":
        inj_changed = xinjections[parameters[nom]["filter"]].copy()
        inj_changed["p0"] = inj_changed["p0"] + 100 * inj_changed["p0"] / inj_changed["p0"].sum()
        network.update_dangling_lines(inj_changed[["p0"]])
        res = pp.loadflow.run_dc(network, parameters=parameters_lf)
        flows[nom] = (get_flows(network) - initial_flows)     
        network.update_dangling_lines(xinjections.loc[parameters[nom]["filter"], ["p0"]])


# ### PTDF matrix generation

# In[56]:


PTDF = pd.DataFrame(flows) / 100
PTDF.head()


# ### All generators started proportionnaly to their Pmax 

# ## Get yearly values

# ### Load data for ES, PT and FR

# In[57]:


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


# In[58]:


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

# In[59]:


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


# In[60]:


for tso in TSOS:
    generation[tso + "_Gen_Total"] = generation.filter(regex=tso + ".*Aggregated").sum(axis=1) - generation.filter(regex=tso + ".*Consumption").sum(axis=1)
generation.filter(like="Total")


# In[61]:


generation.head()


# In[62]:


generation = generation.rename(columns=lambda x: x.replace(" - Actual Aggregated [MW]", "").strip())
generation.head()


# In[63]:


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

# In[64]:


links = ["FR_ES", "FR_BE", "FR_CH", "FR_DE", "FR_IT", "FR_UK", "ES_PT"]
crossborders_flows = pd.concat(
    [
        pd.read_csv(Path("./data") / f"{link}_Cross-Border Physical Flow_202101010000-202201010000.csv", sep=",", index_col=0).reset_index(drop=True)
        for link in links
    ], axis=1
)
crossborders_flows = crossborders_flows.rename(columns={col: re.sub(r"[a-zA-Z ]*\((.*)\) > [a-zA-Z ]*\((.*)\) \[MW\]", r'\1_\2', col) for col in crossborders_flows.columns})


# In[65]:


for link in links:
    crossborders_flows[link] = crossborders_flows[link] - crossborders_flows[link.split("_")[1] + "_" + link.split("_")[0]]
crossborders_flows = crossborders_flows[links]
crossborders_flows


# ### Check that balances are correct for each TSO

# #### Calculate the balance using load and generation#

# In[66]:


balance_generation_load = {}
for tso in TSOS:
    balance_generation_load[tso] = generation[f"{tso}_Gen_Total"] - load[f"{tso}_Load"]


# #### Calculate the balance using exchanges

# In[67]:


balance_exchanges = {
    "ES": - crossborders_flows["FR_ES"] + crossborders_flows["ES_PT"],
    "PT": -crossborders_flows["ES_PT"],
    "FR":  (crossborders_flows["FR_ES"] + crossborders_flows["FR_BE"] + crossborders_flows["FR_DE"] + crossborders_flows["FR_CH"]+ crossborders_flows["FR_IT"] + crossborders_flows["FR_UK"])
}


# #### Comparison of balances

# In[68]:


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

# In[69]:


for tso in TSOS:
    load[f"{tso}_Load"] -= (balance_exchanges[tso] - balance_generation_load[tso])
    balance_generation_load[tso] = generation[f"{tso}_Gen_Total"] - load[f"{tso}_Load"]
    print(tso)
    print((balance_exchanges[tso] - balance_generation_load[tso]).mean())


# ### Build TSO data

# In[70]:


generation["FR_Other_Gen"] = generation["FR_Gen_Total"] - generation["FR_Nuclear"]
generation["ES_Other_Gen"] = generation["ES_Gen_Total"] - generation["ES_Wind Onshore"]


# In[71]:


tso_data = pd.concat([load, crossborders_flows, generation["FR_Nuclear"], generation["ES_Wind Onshore"], generation["FR_Other_Gen"], generation["ES_Other_Gen"], generation["PT_Gen_Total"] ], axis=1)
tso_data["ES_MA"] = 0
tso_data["FR_IE"] = 0
tso_data


# ## Calcultate yearly flows

# In[72]:


flows = initial_flows + (tso_data[PTDF.columns] - initial_tso_data).dot(PTDF.T)
flows


# In[73]:


flows["_18d6873c-e075-524b-8ad0-c6a4aa514a04"]


# In[74]:


flows["_aa3e5611-4a1c-5c98-8d7f-cc5892751ee0"]


# In[75]:


flows["_a8e44e5c-a57a-5c95-9011-fb724230da32"]


# In[76]:


flows["_64511498-394f-5f79-a185-a52044271a8c"]


# In[77]:


PTDF.loc["SLACK_LINE"]


# In[78]:


flows["_363fc03a-2307-5959-20f7-38c771355440"]


# In[79]:


flows["_0b3075b2-7980-988a-224b-407447636928"]


# In[80]:


flows["_0a3cbdb0-cd71-52b0-b93d-cb48c9fea3e2"]


# In[81]:


flows["_2e81de07-4c22-5aa1-9683-5e51b054f7f8"]


# # Calculate FR-ES transfer capacity in **N condition**

# ## Calculate sensitivity to exchanges between FR-ES

# In[82]:


powershift_sensitivity = (PTDF['FR_Load'] - PTDF['ES_Load'])
powershift_sensitivity = powershift_sensitivity[powershift_sensitivity.abs() > 0.05]
powershift_sensitivity_with_linenames = powershift_sensitivity.droplevel(0)


# ## Extract ratings from the models

# In[83]:


monitored_lines = ["HERNANI_XHE_AR11_1_400", "XAR_AR21_DESF.ARK_1_220", "BIESCAS_XBI_PR21_1_220", "VIC_XVI_BA11_1_400"]
tie_lines = xinjections.merge(voltage_levels, left_on="voltage_level_id", right_index=True, how="left", suffixes=("", "vl"))
ratings = network.get_operational_limits()
patl_ratings = ratings[ratings["acceptable_duration"] == -1]
line_ratings = {}
for monitored_line in monitored_lines:
    rdfid = tie_lines[tie_lines["name"] == monitored_line].index
    nominal_voltage = tie_lines.loc[tie_lines["name"] == monitored_line, "nominal_v"].to_numpy()[0]
    line_ratings[monitored_line] = patl_ratings.loc[rdfid, "value"].to_numpy()[0] * nominal_voltage * math.sqrt(3) / 1_000
line_ratings


# ## Calculate maximum powershift in N condition

# In[84]:


from ortools.linear_solver import pywraplp

N_RATING_SECURITY_MARGIN = 0.9

# Get the flows for monitored line only
flows_n = flows.droplevel(0, axis=1)[monitored_lines]

ratings = N_RATING_SECURITY_MARGIN * pd.DataFrame(line_ratings, index=flows_n.index)

solutions = []
# Calculate lower and upper bound to feed the optimizer
lower_bound = pd.DataFrame(0, columns=monitored_lines, index=flows_n.index)
upper_bound = pd.DataFrame(0, columns=monitored_lines, index=flows_n.index)
for monitored_line in tqdm(monitored_lines, desc="calculate upper/lower bounds"):
    lower_bound[monitored_line] = -ratings[monitored_line] - flows_n[monitored_line]
    upper_bound[monitored_line] = ratings[monitored_line] - flows_n[monitored_line]
    
# For each PiT, calculate the highest (lowest) powershift that respect all the N constraints
for pit in tqdm(range(len(flows)), desc="powershift calculation"):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # Variable used to optimize the cost function: only powershift here
    powershift = solver.NumVar(-solver.infinity(), solver.infinity(), 'powershift')

    # Build the constraints
    constraints = {}
    for iconstraint, monitored_line in enumerate(monitored_lines):
        # Define each constraint: first, set lower bounds and upper bounds
        constraints[iconstraint] = solver.Constraint(float(lower_bound.loc[pit, monitored_line]), float(upper_bound.loc[pit, monitored_line]))
        # Set how the constraint changes regarding the powershift
        constraints[iconstraint].SetCoefficient(powershift, float(powershift_sensitivity_with_linenames[monitored_line]))
    # Find the maximum Powershift A->B    
    solver.Maximize(powershift)
    status = solver.Solve()
    solution_max = powershift.solution_value()
    # Find the maximum Powershift B->A
    solver.Minimize(powershift)
    status = solver.Solve()
    solution_min = powershift.solution_value()
    solutions.append([solution_max, solution_min])


# In[85]:


flows.droplevel(0, axis=1)[monitored_lines].head()


# In[86]:


pd.DataFrame(solutions, columns=["max", "min"]).plot()


# # Calculate FR-ES transfer capacity in **N-1 condition**

# In[87]:


network.get_tie_lines().head()


# In[88]:


tie_lines_id = xinjections.loc[xinjections["name"].isin(("HERNANI_XHE_AR11_1_400", "XAR_AR21_DESF.ARK_1_220", "BIESCAS_XBI_PR21_1_220", "VIC_XVI_BA11_1_400")), "tie_line_id"]
tie_lines = network.get_tie_lines()
tie_lines.loc[tie_lines_id]


# In[89]:


monitored_lines_for_otdf = tie_lines.loc[tie_lines_id, "name"].to_dict()
monitored_lines_for_otdf


# In[90]:


lines_id_dict = initial_flows.reset_index(level=1)["name"].to_dict()


# In[91]:


contingencies = powershift_sensitivity[powershift_sensitivity.abs() > 0.05].droplevel(1).index


# In[92]:


sa = pp.sensitivity.create_dc_analysis()
sa.add_branch_flow_factor_matrix(
    list(monitored_lines_for_otdf), [loads.index[0]], "otdf"
)
for contingency in contingencies:
    sa.add_single_element_contingency(contingency)

sa_result = sa.run(network, parameters_lf)
n_1 = pd.concat(
    [
        sa_result.get_reference_matrix("otdf", contingency)
        for contingency in contingencies
    ],
)
n_1.index = contingencies


# In[93]:


n_1.sub(sa_result.get_reference_matrix("otdf").squeeze(), axis=1).rename(index=lines_id_dict)


# In[94]:


otdf_matrix = (
        n_1.sub(sa_result.get_reference_matrix("otdf").squeeze(), axis=1).rename(index=lines_id_dict)
        .divide(initial_flows.loc[contingencies].droplevel(0), axis=0)
        .fillna(0)
    )
otdf_matrix


# In[95]:


# Tie-lines are made of dangling_line1 + dangling_line2
# Direction of flows of tie-line is the same as dangling_line1, but in the other direction for dangling_line2
for col in otdf_matrix.columns:
    if " + " in col:
        otdf_matrix[tie_lines.at[col, "dangling_line1_id"]] = otdf_matrix[col]
        otdf_matrix[tie_lines.at[col, "dangling_line2_id"]] = - otdf_matrix[col]
otdf_matrix = otdf_matrix.rename(columns=lines_id_dict)
otdf_matrix = otdf_matrix[monitored_lines]


# In[96]:


otdf_matrix


# In[97]:


flows = flows.droplevel(axis=1, level=0)


# In[98]:


contingencies = powershift_sensitivity_with_linenames.index


# In[99]:


from collections import defaultdict


# In[100]:


from ortools.linear_solver import pywraplp
import itertools

N_RATING_SECURITY_MARGIN = 0.9
n_ratings = N_RATING_SECURITY_MARGIN * pd.DataFrame(line_ratings, index=flows.index)
n_1_ratings = pd.DataFrame(line_ratings, index=flows.index)

cbcos_idx = pd.MultiIndex.from_product(
    [list(monitored_lines), contingencies]
)
upper_bounds = pd.DataFrame(0, columns=cbcos_idx, index=flows.index)
lower_bounds = pd.DataFrame(0, columns=cbcos_idx, index=flows.index)

sensitivities = pd.Series(0, index=cbcos_idx)
print("Preparing the optimization problem...")
for monitored_line in tqdm(monitored_lines):
    for contingency in contingencies:
        cbco = (monitored_line, contingency)
        sensitivity = (
            powershift_sensitivity_with_linenames.loc[monitored_line]
            + otdf_matrix.loc[contingency, monitored_line]
            * powershift_sensitivity_with_linenames.loc[contingency]
        )
        sensitivities[(monitored_line, contingency)] = sensitivity
        if monitored_line == contingency:
            # N condition
            upper_bounds[(monitored_line, contingency)] = (
                n_ratings[monitored_line] - flows[monitored_line]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_ratings[monitored_line] - flows[monitored_line]
            )
        else:
            # N-1 condition
            upper_bounds[(monitored_line, contingency)] = (
                n_1_ratings[monitored_line]
                - flows[monitored_line]
                - otdf_matrix.loc[contingency, monitored_line] * flows[contingency]
            )
            lower_bounds[(monitored_line, contingency)] = (
                -n_1_ratings[monitored_line]
                - flows[monitored_line]
                - otdf_matrix.loc[contingency, monitored_line] * flows[contingency]
            )

powershift_result = defaultdict(list)
pst_hvdc_result = defaultdict(list)
critical_branches = defaultdict(list)
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
            powershift, float(sensitivities[iconstraint])
        )

    # Max powershift in A->B direction and min powershift in B->A direction
    for direction in ["max", "min"]:
        if direction == "max":
            solver.Maximize(powershift)
        elif direction == "min":
            solver.Minimize(powershift)
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

        powershift_result[direction].append(powershift_sol)
        critical_branches[direction].append(critical_branch)



# In[101]:


pd.DataFrame(powershift_result).head()


# In[102]:


pd.DataFrame(powershift_result).plot()

