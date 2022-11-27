from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair, _Element, ElectricityVector
from .production_base import Production


class ElementConverter(Production):
    def __init__(self,
                 element_1,
                 element_2,
                 direction,
                 name=None
                 ):
        """
        ElementConverter components convert an element into another.

        This component declares the following exported decision variables:

        * X_P, binary.
          Whether the component is used by the hub.
        * SP_P, continuous, in kW.
          The maximum capacity of the component.
        * For all t, for all element e, F_P(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).
        * For all t, Q_P(t), continuous, in kW.
          The reference power related to the component.
          This power is a lower bound of SP_P.
          There exists one element e such that Q_P(t) = F_P(e, t) or Q_P(t) = - F_P(e, t).
          For this component, e is either `element_1` or `element_2` depending on the `direction` attribute.

        This component does not declare any KPI.


        Parameters
        ----------
        element_1 : ElectricityVector, FuelVector, ThermalVectorPair, ThermalVector
        element_2 : ElectricityVector, FuelVector, ThermalVectorPair, ThermalVector
        direction : {'produced', 'consumed', 'both'}
            Related to `element_1`.

            * 'produced': a flow of `element_1` is produced, a flow of `element_2` is consumed
            * 'consumed': a flow of `element_2` is produced, a flow of `element_1` is consumed
            * 'both': depending on the time step t, a flow of `element_1` is produced or consumed
               In this mode, decision variable 'Q_P' has no upper bound.

        name : str, optional

        Notes
        -----
        No `properties` argument must be specified for this component:

        * "LB max output power (kW)" is set to 0 kW
        * "UB max output power (kW)" is set to 1 GW
        * no costs are associated with the component

        """
        properties = {"LB max output power (kW)": 0, "UB max output power (kW)": 1000000}
        if name is None:
            name = f"Converter..{element_1}..{element_2}"
        super().__init__(properties, name, given_sizing=None,
                         units_number_ub=1, units_number_lb=1, eco_count=False)
        self.direction = direction
        self._add_used_element(element_1, "element_1", _Element)
        self._add_used_element(element_2, "element_2", _Element)

    @property
    def element_1(self):
        """
        ElectricityVector, FuelVector, ThermalVectorPair, ThermalVector

        """
        return self._element_1

    @property
    def element_2(self):
        """
        ElectricityVector, FuelVector, ThermalVectorPair, ThermalVector

        """
        return self._element_2

    @property
    def direction(self):
        """
        Related to `element_1`.

        * 'produced': a flow of `element_1` is produced, a flow of `element_2` is consumed
        * 'consumed': a flow of `element_2` is produced, a flow of `element_1` is consumed
        * 'both': depending on the time step t, a flow of `element_1` is produced or consumed
           In this mode, decision variable 'Q_P' has no upper bound.

        {'produced', 'consumed', 'both'}

        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        assert direction in ["produced", "consumed", "both"]
        self._direction = direction

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)



        if self.direction == "consumed":
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(0) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)] == model_data.vars["F_P"][(hub, self._element_1, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))

        elif self.direction == "produced":
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)] == model_data.vars["F_P"][(hub, self._element_2, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))

        else:
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)] >= model_data.vars["F_P"][(hub, self._element_1, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production 1", [hub], [self], TS._t[:-1]))
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)] >= - model_data.vars["F_P"][(hub, self._element_1, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production 2", [hub], [self], TS._t[:-1]))

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_P"][(hub, self._element_1, self, t)]
             == - model_data.vars["F_P"][(hub, self._element_2, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))







class Pump(Production):

    def __init__(self,
                 element_1,
                 element_2,
                 energy_drive,
                 properties,
                 direction = "both",
                 pump_consumption=0.005,
                 given_sizing=None,
                 name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True
                 ):

        """
        Pump components convert a ThermalVectorPair into another ThermalVectorPair.
        Electricity is consumed in proportion of the pumped power.

        This component declares the following exported decision variables:

        * X_P, binary.
          Whether the component is used by the hub.
        * SP_P, continuous, in kW.
          The maximum capacity of the component. Defines the investment costs.
        * For all t, for all element e, F_P(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).
        * For all t, Q_P(t), continuous, in kW.
          The reference power related to the component. Defines the variable cost.
          This power is a lower bound of SP_P.
          There exists one element e such that Q_P(t) = F_P(e, t) or Q_P(t) = - F_P(e, t).
          For this component, e is `energy_drive`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.

        Parameters
        ----------
        element_1 : ThermalVectorPair
        element_2 : ThermalVectorPair
        energy_drive : ElectricityVector
        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/kW)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        direction : {'produced', 'consumed', 'both'}, optional, default 'both'
            Related to `element_1`.

            * 'produced': a flow of `element_1` is produced, a flow of `element_2` is consumed
            * 'consumed': a flow of `element_2` is produced, a flow of `element_1` is consumed
            * 'both': depending on the time step t, a flow of `element_1` is produced or consumed
               In this mode, decision variable 'Q_P' has no upper bound.
               In 'Eco' optimization, the component cost constrains `Q_P`.

        pump_consumption : int, float or numpy.ndarray, optional, default 0.005
            Instantaneous electrical consumption of the component in proportion of the pumped power.
        name : str, optional

        """
        if name is None:
            name = f"Pump..{element_1}..{element_2}"
        super().__init__(properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self.direction = direction
        self._add_used_element(element_1, "element_1", _ThermalVectorPair)
        self._add_used_element(element_2, "element_2", _ThermalVectorPair)
        bool_1 = element_1.is_cooled
        bool_2 = element_2.is_cooled
        assert (bool_1 and bool_2) or (not (bool_1 or bool_2)), \
            f"{self}: 'element_1' and 'element_2' must be both warmed up or cooled down."
        self._add_used_element(energy_drive, "energy_drive", ElectricityVector)
        self.pump_consumption = pump_consumption



    @property
    def pump_consumption(self):
        return self._pump_consumption


    @pump_consumption.setter
    def pump_consumption(self, pump_consumption):
        DataAccessors.type_checker(pump_consumption, self, "pump_consumption", "numeric")
        self._pump_consumption = pump_consumption

    @property
    def energy_drive(self):
        """
        ElectricityVector

        """
        return self.energy_drive


    @property
    def element_1(self):
        """
        ThermalVectorPair

        """
        return self._element_1

    @property
    def element_2(self):
        """
        ThermalVectorPair

        """
        return self._element_2

    @property
    def direction(self):
        """
        Related to `element_1`.

        * 'produced': a flow of `element_1` is produced, a flow of `element_2` is consumed
        * 'consumed': a flow of `element_2` is produced, a flow of `element_1` is consumed
        * 'both': depending on the time step t, a flow of `element_1` is produced or consumed
           In this mode, decision variable 'Q_P' has no upper bound.

        {'produced', 'consumed', 'both'}

        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        assert direction in ["produced", "consumed", "both"]
        self._direction = direction


    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)

        [model_data.vars["F_P"][(hub, self._energy_drive, self, t)].set_lb(0) for t in TS._t[:-1]]

        if self.direction == "consumed":
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(0) for t in TS._t[:-1]]

            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                            (model_data.vars["F_P"][(hub, self._energy_drive, self, TS._t[ind_t])]
                              == model_data.vars["F_P"][(hub, self._element_1, self, TS._t[ind_t])]
                             * DataAccessors.get2(self._pump_consumption, TS._t[ind_t], TS._t[ind_t+1])
                              for ind_t in range(len(TS._t[:-1]))),
                              names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))


        elif self.direction == "produced":
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                                (model_data.vars["F_P"][(hub, self._energy_drive, self, TS._t[ind_t])]
                                 == model_data.vars["F_P"][(hub, self._element_2, self, TS._t[ind_t])]
                                 * DataAccessors.get2(self._pump_consumption, TS._t[ind_t], TS._t[ind_t+1])
                              for ind_t in range(len(TS._t[:-1]))),
                                      names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))

        else:
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_1, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._element_2, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                            (model_data.vars["F_P"][(hub, self._energy_drive, self, TS._t[ind_t])]
                              >= - model_data.vars["F_P"][(hub, self._element_1, self, TS._t[ind_t])]
                             * DataAccessors.get2(self._pump_consumption, TS._t[ind_t], TS._t[ind_t + 1])
                              for ind_t in range(len(TS._t[:-1]))),
                              names=NamesFormatter.fmt("Power balance 1", [hub], [self], TS._t[:-1]))
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                            (model_data.vars["F_P"][(hub, self._energy_drive, self, TS._t[ind_t])]
                              >= model_data.vars["F_P"][(hub, self._element_1, self, TS._t[ind_t])]
                             * DataAccessors.get2(self._pump_consumption, TS._t[ind_t], TS._t[ind_t + 1])
                              for ind_t in range(len(TS._t[:-1]))),
                              names=NamesFormatter.fmt("Power balance 2", [hub], [self], TS._t[:-1]))


        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["Q_P"][(hub, self, t)] == model_data.vars["F_P"][(hub, self._energy_drive, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))


        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_P"][(hub, self._element_1, self, t)]
             == - model_data.vars["F_P"][(hub, self._element_2, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Thermal balance", [hub], [self], TS._t[:-1]))