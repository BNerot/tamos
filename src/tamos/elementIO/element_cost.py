# -*- coding: utf-8 -*-

from tamos.component import MetaComponent
from tamos.data_IO.data_IO import DataAccessors, NamesFormatter

class Cost(MetaComponent):
    def __init__(self, cost=None, carbon_cost=None, name=None):
        """
        Defines the cost of buying energy from or selling energy to the outside of the energy system.
        A Cost instance must be passed to a Grid or Load instance to be effective.
        All costs are defined relatively to the flow of the `element` attribute of the Grid or Load instance.

        Cost instances declare the following KPIs:

        * `COST_element`
          In euros.
          Contributes to the "Eco" objective function. Related to the `cost` attribute.
        * `Carbon tax`
          In euros.
          Contributes to the "Eco" objective function. Related to the `carbon_cost` attribute.

        Parameters
        ----------
        cost : int, float or numpy.ndarray, optional
           Cost associated with a positive flow of `element`.
           In euros/kWh.
           Usually negative.
        carbon_cost : int, float or numpy.ndarray, optional
           Cost associated with positive CO2 emissions regarding the flow of 'element'.
           In euros/kgEqCO2.
           Usually positive.
        name : str, optional

        Examples
        --------
        >>> grid=  Grid(element=electricity, emissions=0.3)
        >>> cost = Cost(cost=-0.2, carbon_cost=0.1)
        >>> grid.element_cost = cost

        The energy system paies 0.2€ to buy 1 kWh of `electricity` from `grid`,
        added to 0.3 * 0.1 = 0.03€ paid as a carbon tax.

        """
        if name is None:
            name = f"Cost"
        super().__init__(name)
        self._cost, self._carbon_cost = None, None
        self.cost = cost
        self.carbon_cost = carbon_cost

    @property
    def cost(self):
        """
        Cost associated with a positive flow of `element`.
        In euros/kWh.
        Usually negative.
        int, float or numpy.ndarray

        """
        return self._cost

    @cost.setter
    def cost(self, cost):
        if cost is not None:
            DataAccessors.type_checker(cost, self, "cost", "numeric")
        self._cost = cost

    @property
    def carbon_cost(self):
        """
        Cost associated with positive CO2 emissions regarding the flow of 'element'.
        In euros/kgEqCO2.
        Usually positive.
        int, float or numpy.ndarray

        """
        return self._carbon_cost

    @carbon_cost.setter
    def carbon_cost(self, carbon_cost):
        if carbon_cost is not None:
            DataAccessors.type_checker(carbon_cost, self, "carbon_cost", "numeric")
        self._carbon_cost = carbon_cost

    def _declare_KPIs(self, model_data, hub, element_IO, TS):
        values = {}
        args = []
        if self.cost is not None:
            costs = [DataAccessors.get2(self.cost, TS._t[ind_t], TS._t[ind_t + 1])
                     for ind_t in range(len(TS._t[:-1]))]
            args += [("COST_element", "Eco", costs)]

        if self.carbon_cost is not None:
            if element_IO.emissions is None:
                print(f"Warning: {self}: Defining a carbon cost has no effect "
                      f"if `emissions` attribute of {element_IO} is undefined.")
            else:
                taxes = [DataAccessors.get2(self.carbon_cost*element_IO.emissions, TS._t[ind_t], TS._t[ind_t + 1])
                         for ind_t in range(len(TS._t[:-1]))]
                args += [("Carbon tax", "Eco", taxes)]

        for base_name, kind, data in args:
            values[f"{base_name}"] = model_data.mdl.sum(
                model_data.vars["F_EXT"][(hub, element_IO, TS._t[ind_t])]
                * (data[ind_t]
                   * TS._dt[ind_t])
                for ind_t in range(len(TS._t[:-1]))) * TS._step_value

            value, CAPEX__, OPEX__ = element_IO.compute_actualized_cost(0,
                                                                  values[base_name],
                                                                  TS._system_lifetime
                                                                  )
            for name__, value, used_in_solving in [(f"{base_name} - CAPEX", CAPEX__, False),
                                                   (f"{base_name} - OPEX", OPEX__, False),
                                                   (f"{base_name}", value, True),
                                                   ]:
                name = NamesFormatter.fmt_light(name__, hub, element_IO, KPI=True)
                model_data.add_KPI_use(name__, ("Hub", "ElementIO"))
                kpi = model_data.mdl.add_kpi(value, publish_name=name)
                model_data.KPIs[hub][element_IO][self] += [model_data.KPI_wrapper(kpi, kind, used_in_solving)]
