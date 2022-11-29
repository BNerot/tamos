from numpy.random import RandomState
rs = RandomState(1234)

from tamos import Hub, MILPModel, TimeSettings, use_name_in_MILP
from tamos.data_IO import ResultsExport
from tamos.elementIO import Cost, Grid, Load
from tamos.element import ThermalVector, ElectricityVector, fetch_TVP
from tamos.production import ElectricHeater
from tamos.storage import Thermocline

use_name_in_MILP(True)

"""
A space heating demand is met by an electric boiler and a space heating storage
"""

# Declaration of energy vectors.
electricity = ElectricityVector(name="Electricity")

in_TV=ThermalVector(temperature=60+273, name="Entering hot water")
out_TV=ThermalVector(temperature=30+273, name="Exiting cold water")

space_heating = fetch_TVP(
    in_TV=in_TV,
    out_TV=out_TV,
    name="Thermal emitter flow"
)


# The energy system comprises only one hub.
hub = Hub(name="Heat consumer")

# This hub has a space heating demand that must be satisfied.
load = 10 * (rs.random(size=8760) + 3)
load[2000:-2000] = 0
space_heating_demand = Load(load=load,
                            element=space_heating,
                            name="Heat demand")
hub.change_components(space_heating_demand, add=True)
hub.components_assemblies = [(1, 1, space_heating_demand)]



# This demand is met by two components whose sizing must be determined:
# - an electric boiler;
# - a short-term thermal energy storage (max capacity: 100 kg);
electric_boiler = ElectricHeater(energy_source=electricity,
                           energy_sink=~space_heating,
                           properties={"CAPEX (EUR/kW)": 643.8,
                                        "OPEX (%CAPEX)": 0.05,
                                        "Variable OPEX (EUR/MWh)": 0,
                                        "LB max output power (kW)": 0,
                                        "UB max output power (kW)": 100,
                                        "Discount rate (%)": 3.5,
                                        "Lifetime": 30},
                           name="Electric boiler")

heat_storage = Thermocline(stored_TVP=space_heating,
                           properties= {"CAPEX energy (EUR/kg)" : 14280e-3,
                                        "OPEX energy (%CAPEX)" : 4.07,
                                        "Charge/discharge delay (h)" : 1,
                                        "LB max energy (kg)" : 0,
                                        "UB max energy (kg)" : 1000,
                                        "Energy conservation (/h)" : 0.979,
                                        "Discount rate (%)" : 3.5,
                                        "Lifetime" : 30},
                           name="Heat storage")
hub.change_components([electric_boiler, heat_storage], add=True)


# The boiler consumes electricity.
# The cost of electricity (negative) is as seen by the electricity grid,
# the opposite of this cost (positive) is as seen by the hub. The same rule applies for CO2 emissions.
cost = Cost(cost=-(0.15 + (rs.random(size=8760)-0.5) * 0.05))

electricity_grid = Grid(
                        element=electricity,
                        element_cost=cost,
                        emissions=-0.2
                        )
hub.change_components(electricity_grid, add=True)



# The 8760 first elements of time series (space heating load, electricity cost) are considered.
# Each element is 1 hour length.
# Amortization of economic values is done on a 40 years basis.
# Every 1 nth element in [0, 8759] is considered, i.e. all elements.
ts = TimeSettings(8760, 1, 40)
ts.add_regular(1)

energy_system = MILPModel(hubs=[hub],
                          time_settings=ts,
                          name="System model 0_05")

# The optimization+sizing problem is solved on an economic basis using Cplex Solver.
energy_system.declare_variables()
energy_system.declare_constraints_and_KPIs()
energy_system.solve(kind="Eco")

# Results are first loaded and then written on disk.
results = ResultsExport(MILPModel=energy_system, get_LP=True)
results.write_all()

# One can have a look at the main decision variables.
try:
    print(results._var_dataframes["SP_P"])
    print(results._var_dataframes["SE_S"])
except:
    pass
print(
    results.solution_summary[["Objective value", "Eco", "CO2", "Exergy"]].astype(int).to_dict()
)

