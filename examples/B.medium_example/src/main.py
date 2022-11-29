from numpy import sign
from numpy.random import RandomState
rs = RandomState(1234)

from pathlib import Path

from tamos import Hub, MILPModel, TimeSettings, allow_duplicated_names, InterfaceMask, use_name_in_MILP
from tamos.data_IO import ResultsBatch, ResultsExport
from tamos.elementIO import Cost, Grid, Load
from tamos.element import fetch_TVP, ThermalVector, FuelVector, get_dead_state_temperature
from tamos.network import ThermalNetwork
from tamos.production import GasBoiler, HeatExchanger
from tamos.storage import Thermocline

T_ref = get_dead_state_temperature()
use_name_in_MILP(True)

"""
Some consumer hubs have a heating demand. Each of them may install a gas boiler, or get heat delivered from 
a heating network. The heating network is powered by an external heat source and a heat storage in a production hub.
CO2 emissions are minimized. A second optimization is ran to get real costs.
"""

# Declaration of energy vectors.
natural_gas = FuelVector(
                        exergy_factor=1 - T_ref / FuelVector.typical_flame_temperature["Natural gas"],
                        name="Natural gas"
                        )

space_heating = fetch_TVP(
    in_TV=ThermalVector(temperature=60+273, name="Entering hot water"),
    out_TV=ThermalVector(temperature=30+273, name="Exiting cold water"),
    name="Thermal emitter flow"
)

network_carrier = fetch_TVP(
    in_TV=ThermalVector(temperature=90+273, name="Warm pipe"),
    out_TV=ThermalVector(temperature=60+273, name="Cold pipe"),
    name="Network flow"
)

# The energy system is made of four consumer hubs and one production-only hub.
# All thermal demands are different.
hub_producer = Hub(name="P")
hubs_locations = {}
hubs_locations[hub_producer] = (0, 0)

hubs_consumers = []
N = 3			# number of consumer hubs
for hub_number in range(1, N+1):
    hub = Hub(name=f"C{hub_number}")
    hubs_locations[hub] = rs.normal(loc=0, scale=0.01, size=2) * 10
    space_heating_demand = Load(load=rs.random(size=8760) * rs.random() * 5, element=space_heating, name=f"Demand {hub_number}")
    hub.change_components(space_heating_demand, add=True)
    hub.components_assemblies = [(1, 1, space_heating_demand)]
    hubs_consumers.append(hub)



# Hubs might be connected between each others using a district heating network.
heating_network = ThermalNetwork(hubs_locations=hubs_locations,
                                 production_hub=hub_producer,
                                 production_mode="heat",
                                 element=network_carrier,
                                 properties={"CAPEX (EUR/km)": 500,
                                             "OPEX (%CAPEX)": 0.06,
                                             "Discount rate (%)": 3.5,
                                             "Lifetime": 40,
                                             "Variable OPEX (EUR/MWh)": 1.5},
                                 name="Heating network")

# Network connections must be determined given the following constraint:
# energy can flow between two given hubs in one direction only during the operation period.
for hub in hubs_locations.keys():
    heating_network.set_node_status(hub, heating_network.optim_one_way_max)

# Are the network connections as expected?
# ax = heating_network.plot()

# the producer hub can use a gas boiler and heat storage
network_storage = Thermocline(stored_TVP=network_carrier,
                           properties= {"CAPEX energy (EUR/kg)" : 14280e-3,
                                        "OPEX energy (%CAPEX)" : 4.07,
                                        "Charge/discharge delay (h)" : 1,
                                        "LB max energy (kg)" : 0,
                                        "UB max energy (kg)" : 100,
                                        "Energy conservation (/h)" : 0.979,
                                        "Discount rate (%)" : 3.5,
                                        "Lifetime" : 30},
                           name="Storage")
                           
heat_source = Grid(element=~network_carrier,
                   emissions=-(0.2 + rs.random(size=8760) * 0.1))
mask = InterfaceMask(component=heat_source, power_ub=50)
hub_producer.change_components([heat_source, network_storage], add=True)
hub_producer.change_interface_masks([mask], add=True)

# A network heat exchanger is needed in every hubs to benefit from centrally produced heat.
# Individual gas boilers are also considered.

heat_exchanger = HeatExchanger(energy_source=network_carrier,
                               energy_sink=~space_heating,
                               properties={"CAPEX (EUR/kW)": 265,
                                           "OPEX (%CAPEX)": 0,
                                           "Variable OPEX (EUR/MWh)": 1.5,
                                           "LB max output power (kW)": 0,
                                           "UB max output power (kW)": 100,
                                           "Discount rate (%)": 3.5,
                                           "Lifetime":40},
                               name="Heat exchanger")
gas_boiler_consumer = GasBoiler(energy_source=natural_gas,
                               energy_sink=~space_heating,
                               properties={"CAPEX (EUR/kW)": 60,
                                           "OPEX (%CAPEX)": 3.25,
                                           "Variable OPEX (EUR/MWh)": 1.1,
                                           "LB max output power (kW)": 0,
                                           "UB max output power (kW)": 100,
                                           "Discount rate (%)": 3.5,
                                           "Lifetime": 25},
                               name="Gas boiler")

for hub in hubs_consumers:
    hub.change_components([heat_exchanger, gas_boiler_consumer], add=True)


# All hubs must be able to import natural gas.
# In this example, the CO2 content of natural gas varies in time.
natural_gas_grid = Grid(
                        element=natural_gas,
                        element_cost=Cost(cost=-0.1),
                        emissions=-(0.2 + rs.random(size=8760) * 0.1)
                        )
[hub.change_components(natural_gas_grid, add=True) for hub in hubs_consumers]



# What are the components of each consumer hub now? e.g. hub 0
# hubs_consumers[0].describe()

ts = TimeSettings(8760, 1, 40)
ts.add_regular(1)

energy_system = MILPModel(hubs=hubs_consumers + [hub_producer],
                          time_settings=ts,
                          name=f"System model {N}")
# One consumer cannot connect to the network.
energy_system.components_assemblies = [(0, N-1, heat_exchanger)]
# The optimization+sizing problem is solved minimizing CO2 emissions. Cplex solver is used.
energy_system.declare_variables()
energy_system.declare_constraints_and_KPIs()
energy_system.solve(kind="CO2", threads=3)
results = ResultsExport(MILPModel=energy_system)

# In CO2 optimization, costs are not constrained. One way to get real costs is to run the model again in Economic optim
# using an epsilon constraint on CO2. A tolerance parameter might be used.
# Notes:
# - the two solutions might differ significantly.
# - the same can be achieved using the AdvSolve class.

CO2_emissions = results.solution_summary["CO2"]
energy_system.declare_max_KPI_constraint("CO2", (1 + sign(CO2_emissions) * 1e-6) * CO2_emissions)
energy_system.solve(kind="Eco", threads=3, MIP_gap=5e-3)
results = ResultsExport(MILPModel=energy_system)
results.write_all()

# Advanced results analysis is done using the dedicated Jupyter Notebook.
# An export as a ResultsBatch instance is required.
RB = ResultsBatch.create_batch_from_binaries(working_dir=Path("../results"),
                                             results_exports=[results],
                                             name=f"Batch {energy_system.name}")

RB.dump_object()
RB.write_all()
