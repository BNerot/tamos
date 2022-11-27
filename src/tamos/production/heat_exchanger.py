from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import ThermalVector, _ThermalVectorPair
from .production_base import Production


class HeatExchanger(Production):
    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties, given_sizing=None, efficiency=0.95, name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        HeatExchanger components convert heat in a passive way.

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
          For this component, e is `energy_sink`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_source : ThermalVectorPair, ThermalVector
            Element that gives thermal energy.
            Must be cooled down if ThermalVectorPair.
        energy_sink : ThermalVectorPair, ThermalVector
            Element that receives thermal energy.
            Must be warmed up if ThermalVectorPair.
        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/kW)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        given_sizing : int or float, optional
            The maximum capacity of the component, in kW.
            Relates to decision variable 'SP_P'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        efficiency: int, float or numpy.ndarray, optional, default 0.95
            The amount of energy received by `energy_sink` for a unit of energy from `energy_source`.
        name : str, optional
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        """
        if name is None:
            name = "HX..{!r}..{!r}".format(energy_source, energy_sink)
        super().__init__(properties, name, given_sizing, units_number_ub, units_number_lb, eco_count)
        self._add_used_element(energy_source, "energy_source", (ThermalVector, _ThermalVectorPair))
        self._add_used_element(energy_sink, "energy_sink", (ThermalVector, _ThermalVectorPair))
        if isinstance(energy_source, _ThermalVectorPair):
            assert energy_source.is_cooled, f"{self}: 'energy_source' must be cooled down. consider using ~({energy_source})."
        if isinstance(energy_sink, _ThermalVectorPair):
            assert energy_sink.is_warmed, f"{self}: 'energy_sink' must be warmed up. consider using ~({energy_sink})."

        self.efficiency = efficiency

    @property
    def energy_source(self):
        """
        Element that gives thermal energy.
        Must be cooled down if ThermalVectorPair.
        ThermalVectorPair, ThermalVector

        """
        return self._energy_source

    @property
    def energy_sink(self):
        """
        Element that receives thermal energy.
        Must be warmed up if ThermalVectorPair.
        ThermalVectorPair, ThermalVector

        """
        return self._energy_sink

    @property
    def efficiency(self):
        """
        The amount of energy received by `energy_sink` for a unit of energy from `energy_source`.
        int, float or numpy.ndarray

        """
        return self._efficiency

    @efficiency.setter
    def efficiency(self, efficiency):
        DataAccessors.type_checker(efficiency, self, "efficiency", "numeric")
        self._efficiency = efficiency

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)

        [model_data.vars["F_P"][(hub, self._energy_source, self, t)].set_lb(0) for t in TS._t[:-1]]

        # energy sink balance, depends on sink type
        if isinstance(self._energy_sink, _ThermalVectorPair):
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_lb(0) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_ub(model_data.inf) for t in TS._t[:-1]]

            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["F_P"][(hub, self._energy_sink, self, TS._t[ind_t])]
                 == model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])]
                 * DataAccessors.get2(self.efficiency, TS._t[ind_t], TS._t[ind_t + 1])
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Thermal balance", [hub], [self], TS._t[:-1]))

            model_data.cts[hub][self] += model_data.mdl.add_constraints((model_data.vars["Q_P"][(hub, self, t)]
                                                                         == model_data.vars["F_P"][
                                                                             (hub, self._energy_sink, self, t)]
                                                                         for t in TS._t[:-1]),
                                                                        names=NamesFormatter.fmt("Ref production",
                                                                                                 [hub], [self],
                                                                                                 TS._t[:-1]))

        else:  # ThermalVector                #for instance, energy_source = infinite lake
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_lb(-model_data.inf) for t in TS._t[:-1]]
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_ub(0) for t in TS._t[:-1]]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink, self, TS._t[ind_t])]
                 == model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])]
                 * DataAccessors.get2(self.efficiency, TS._t[ind_t], TS._t[ind_t + 1])
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Thermal balance", [hub], [self], TS._t[:-1]))

            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)]
                 == - model_data.vars["F_P"][(hub, self._energy_sink, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))
