from tamos.component import Component
from tamos.data_IO.data_IO import DataAccessors, NamesFormatter
from tamos.element import ElectricityVector, FuelVector, ThermalVector, _ThermalVectorPair
from tamos.elementIO.element_cost import Cost


class ElementIO(Component):
    def __init__(self,
                 element,
                 emissions,
                 element_cost,
                 exergy_count,
                 name):
        super().__init__(name, {})
        self.exergy_count = exergy_count
        self.element_cost = element_cost
        self.emissions = emissions
        self._add_used_element(element, "element", (ElectricityVector, FuelVector, ThermalVector, _ThermalVectorPair))

    @property
    def emissions(self):
        """
        Quantity of CO2 associated with a positive power flow of `element`.
        In kgEqCO2/kWh.
        Usually negative.
        int, float or numpy.ndarray

        Examples
        --------
        Examples below are for a Grid component, but Load components behaves similarly.

        >>> natural_gas_grid = Grid(element=natural_gas)
        >>> natural_gas_grid.emissions = - 0.25

        When the Grid component `natural_gas_grid` receives 1 kWh of natural gas from the hub interface
        the net CO2 emissions of the energy system decrease of 0.250 kgEqCO2.
        Conversely, if the energy_system receives 1 kWh of natural gas from `natural_gas_grid`,
        its CO2 emissions increase of 0.250 kgEqCO2.


        >>> thermal_grid = Grid(element=thermal_source)
        >>> thermal_grid.emissions = - 0.05

        `thermal_grid` is such that when the power flow of `thermal_source` is 1 kWh (positive):

        * thermal_source.in_TV enters the grid (i.e leaves the energy system)
          and thermal_source.out_TV leaves the grid (i.e enters the energy_system)
        * the net CO2 emissions of the energy system decrease of 50 gEqCO2.

        """
        return self._emissions

    @emissions.setter
    def emissions(self, emissions):
        if emissions is not None:
            DataAccessors.type_checker(emissions, self, "emissions", "numeric")
        self._emissions = emissions

    @property
    def element_cost(self):
        """
        Cost instance associated with a positive flow of 'element'.
        Cost instance

        """
        return self._element_cost

    @element_cost.setter
    def element_cost(self, element_cost):
        if element_cost is not None:
            DataAccessors.type_checker(element_cost, self, "element_cost", Cost)
        self._element_cost = element_cost

    @property
    def exergy_count(self):
        """
        Whether this instance contributes to the system "Exergy" KPI.
        bool

        """
        return self._exergy_count

    @exergy_count.setter
    def exergy_count(self, exergy_count):
        DataAccessors.type_checker(exergy_count, self, "exergy_count", bool)
        self._exergy_count = exergy_count

    @property
    def element(self):
        """
        Element exchanged between the hub interface and this instance.

        """
        return self._element

    def _declare_variables(self, model_data, hub, TS):
        model_data.add_variable_use("F_EXT", ("Hub", "ElementIO", "Date"))
        model_data.add_variable_use("X_EXT", ("Hub", "ElementIO"))

        model_data.vars["X_EXT"][(hub, self)] = model_data.mdl.binary_var(
            name=NamesFormatter.fmt_light("X_EXT", hub, self))

        model_data.vars["F_EXT"].update(model_data.mdl.continuous_var_dict(((hub, self, t) for t in TS._t[:-1]),
                                                                           lb=- model_data.inf,
                                                                           name=NamesFormatter.fmt("F_EXT", [hub],
                                                                                                   [self], TS._t[:-1])))

    def _declare_constraints(self, model_data, hub, TS):
        model_data.cts[hub][self] = []

        names = NamesFormatter.fmt("Status OFF", [hub], [self], TS._t[:-1])
        model_data.cts[hub][self] += model_data.mdl.add_indicators(
            [model_data.vars["X_EXT"][(hub, self)] for t in TS._t[:-1]],
            [model_data.vars["F_EXT"][(hub, self, t)] == 0 for t in TS._t[:-1]],
            true_values=0,
            names=names)

    def _declare_KPIs(self, model_data, hub, TS):
        model_data.KPIs[hub][self][self] = []
        self._declare_exergy_and_CO2_KPIs(model_data, hub, TS)
        if self.element_cost and (self.element_cost not in model_data.KPIs[hub][self]):
            model_data.KPIs[hub][self][self.element_cost] = []
            self.element_cost._declare_KPIs(model_data, hub, self, TS)

    def _declare_exergy_and_CO2_KPIs(self, model_data, hub, TS):
        values = {}
        args = []
        element = self._element
        if self.emissions is not None:
            sign = 1
            args += [("ElementIO CO2", "CO2", self.emissions, sign)]
        if self.exergy_count:
            sign = 1
            if isinstance(element, _ThermalVectorPair) and element.is_warmed:
                sign = -1
            args += [("ElementIO Exergy", "Exergy", element.exergy_factor, sign)]

        for base_name, kind, data, sign in args:
            values[f"{base_name}"] = sign * model_data.mdl.sum(model_data.vars["F_EXT"][(hub, self, TS._t[ind_t])]
                                                                    * (TS._dt[ind_t]
                                                                    * DataAccessors.get2(data, TS._t[ind_t],
                                                                                         TS._t[ind_t + 1]))
                                                                 for ind_t in range(len(TS._t[:-1]))) * TS._step_value

            model_data.add_KPI_use(base_name, ("Hub", "ElementIO"))
            name = NamesFormatter.fmt_light(base_name, hub, self, KPI=True)
            kpi = model_data.mdl.add_kpi(values[base_name], publish_name=name)
            model_data.KPIs[hub][self][self] += [model_data.KPI_wrapper(kpi, kind, True)]
