from numpy import ndim


from tamos.element import _ThermalVectorPair, ElectricityVector, FuelVector
from .production_base import Production
from ..data_IO.data_IO import NamesFormatter, DataAccessors


class CHP(Production):

    def __init__(self,
                 energy_source,
                 energy_sink_1,
                 energy_sink_2,
                 mode,
                 properties,
                 heat_efficiency_function,
                 given_sizing=None, name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True
                 ):
        """
        CHP components transform a FuelVector element into heat and electricity.

        This model is an adaptation of [1]_.

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
          There exists one element e such that Q_P(t) = F_P(e, t) or Q_P(t) = * F_P(e, t).
          For this component, e is `energy_sink_1`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_source : FuelVector
            Vector that is consumed.
        energy_sink_1 : ElectricityVector
        energy_sink_2 : ThermalVectorPair
            Thermal flow that is warmed up by the CHP.
        mode : {'By-pass', 'Extraction-condensation', 'Back-pressure'}
            Operating mode of the CHP:

            * 'Back-pressure': heat production is proportional to electricity production
            * 'Extraction-condensing': heat and electricity productions are decoupled, with electricity production being favored
            * 'By-pass': heat and electricity productions are decoupled, with heat production being favored

        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "alpha": power-to-heat ratio
              (back-pressure line)
            * "beta": marginal power ratio (increase in power when heat decreases)
              (by-pass mode)
            * "ksi": marginal power ratio (increase in power when heat decreases)
              (extraction-condensation mode)
            * "eta": electrical efficiency
            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/kW)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        heat_efficiency_function : callable f(T)
            [experimental]
            Only for modes 'Back-pressure' and 'Extraction-condensation'.
            T is the temperature of the cold vector of `energy_sink_2`, in Kelvins (K).
            See method `set_efficiency_model` of `GasBoiler` and `BiomassBoiler` classes.
            Setting `heat_efficiency=1` after instanciation makes this attribute unused.
        given_sizing : int or float, optional
            The maximum capacity of the component, in kW.
            Relates to decision variable 'SP_P'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        name : str, optional
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        References
        ----------
        .. [1] DAHL, Magnus, BRUN, Adam et ANDRESEN, Gorm B., 2019.
               Cost sensitivity of optimal sector-coupled district heating production systems.
               Energy. 1 janvier 2019. Vol.166, pp.624‑636. DOI 10.1016/j.energy.2018.10.044.

        """
        if name is None:
            name = "{!r}-CHP..{!r}..{!r}".format(energy_source, energy_sink_1, energy_sink_2)
        super().__init__(properties, name, given_sizing, units_number_ub, units_number_lb, eco_count)
        self._add_used_element(energy_source, "energy_source", FuelVector)
        self._add_used_element(energy_sink_1, "energy_sink_1", ElectricityVector)
        self._add_used_element(energy_sink_2, "energy_sink_2", _ThermalVectorPair)
        assert energy_sink_2.is_warmed, f"{self}: 'energy_sink_2' must be warmed up. consider using ~({energy_sink_2})."

        self.mode = mode
        self.set_heat_efficiency_model(heat_efficiency_function, 2)

    @property
    def energy_source(self):
        """
        Vector that is consumed.

        """
        return self._energy_source

    @property
    def energy_sink_1(self):
        return self._energy_sink_1

    @property
    def energy_sink_2(self):
        return self._energy_sink_2

    @property
    def mode(self):
        """
        Operating mode of the CHP:

        * 'Back-pressure': heat production is proportional to electricity production
        * 'Extraction-condensing': heat and electricity productions are decoupled, with electricity production being favored
        * 'By-pass': heat and electricity productions are decoupled, with heat production being favored

        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        DataAccessors.type_checker(mode, self, "mode", str)
        assert mode in ["By-pass", "Extraction-condensation", "Back-pressure"]
        if (mode == "By-pass") and hasattr(self, "_heat_efficiency"):
            print(f"Warning: {self}: heat_efficiency will not be used anymore.")
        self._mode = mode

    @property
    def heat_efficiency(self):
        """
        [experimental]
        Defines explicitly the heat efficiency of the CHP.
        Only for modes 'Back-pressure' and 'Extraction-condensation'.
        If called, replaces the definition of the efficiency using `set_heat_efficiency_model` (default).
        Setting `heat_efficiency=1` is a safe value.
        int, float or numpy.ndarray

        """
        return self._heat_efficiency

    @heat_efficiency.setter
    def heat_efficiency(self, heat_efficiency):
        if self.mode == "By-pass":
            raise NotImplementedError(f"{self}: no heat_efficiency is taken into account in mode 'By-pass'.")
        else:
            DataAccessors.type_checker(heat_efficiency, self, "heat_efficiency", "numeric")
            self._heat_efficiency = heat_efficiency

    def set_heat_efficiency_model(self, heat_efficiency_function, pinch):
        """
        [experimental]
        Defines the heat efficiency of the CHP.

        Parameters
        ----------
        heat_efficiency_function : callable f(T)
            T is the temperature of the cold vector of `energy_sink`, in Kelvins (K).
        pinch : int, float or numpy.ndarray
            Temperature difference between the flue gases of the boiler and the cold vector of `energy_sink`, in Kelvins (K).

        See method `set_efficiency_model` of `GasBoiler` and `BiomassBoiler` classes.

        """
        DataAccessors.type_checker(pinch, self, "pinch", "numeric")

        T = self._energy_sink_2._in_TV.temperature + pinch
        heat_efficiency = heat_efficiency_function(T)
        if ndim(heat_efficiency) == 0:
            # if T is float then self._heat_efficiency is ndarray of dim 0
            # which can't be handled by DataAccessors.get methods
            heat_efficiency = float(heat_efficiency)
        self.heat_efficiency = heat_efficiency

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        alpha, beta = self._get_property("alpha"), self._get_property("beta")
        eta, ksi = self._get_property("eta"), self._get_property("ksi")

        if self.mode == "Back-pressure":
            # heat_efficiency adaptation
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, TS._t[ind_t])]
                 == alpha * model_data.vars["F_P"][(hub, self._energy_sink_2, self, TS._t[ind_t])]
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Ba-pre line", [hub], [self], TS._t[:-1]))
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])] * eta
                 * DataAccessors.get2(self.heat_efficiency, TS._t[ind_t], TS._t[ind_t + 1])
                 == - model_data.vars["F_P"][(hub, self._energy_sink_1, self, TS._t[ind_t])]
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))

        elif self.mode == "Extraction-condensation":
            # heat_efficiency adaptation
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)]
                 >= alpha * model_data.vars["F_P"][(hub, self._energy_sink_2, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Above ba-pre line", [hub], [self], TS._t[:-1]))

            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)]
                 <= - ksi * model_data.vars["F_P"][(hub, self._energy_sink_2, self, t)]
                 + model_data.vars["SP_P"][(hub, self)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Below ext-cond line", [hub], [self], TS._t[:-1]))

            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])]
                 * eta
                 * DataAccessors.get2(self.heat_efficiency, TS._t[ind_t], TS._t[ind_t + 1])
                 == - model_data.vars["F_P"][(hub, self._energy_sink_1, self, TS._t[ind_t])]
                 + ksi * model_data.vars["F_P"][(hub, self._energy_sink_2, self, TS._t[ind_t])]
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Power balance",
                                         [hub], [self], TS._t[:-1]))
        else:  # self.mode == "By-pass"
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)]
                 <= alpha * model_data.vars["F_P"][(hub, self._energy_sink_2, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Below ba-pre line", [hub], [self], TS._t[:-1]))

            coefficient = 1 / beta + 1 / alpha
            max_thermal_power = coefficient * model_data.vars["SP_P"][(hub, self)]
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)]
                 <= beta * (max_thermal_power - model_data.vars["F_P"][(hub, self._energy_sink_2, self, t)])
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Below by-pass line", [hub], [self], TS._t[:-1]))
            coefficient = alpha / ((beta + alpha) * eta)
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])]
                 == coefficient * (- model_data.vars["F_P"][(hub, self._energy_sink_1, self, TS._t[ind_t])]
                                   + beta
                                   * model_data.vars["F_P"][(hub, self._energy_sink_2, self, TS._t[ind_t])])
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["Q_P"][(hub, self, t)] == - model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))

        [model_data.vars["F_P"][(hub, self._energy_source, self, t)].set_lb(0) for t in TS._t[:-1]]
        [model_data.vars["F_P"][(hub, self._energy_sink_1, self, t)].set_ub(0) for t in TS._t[:-1]]
        [model_data.vars["F_P"][(hub, self._energy_sink_2, self, t)].set_lb(0) for t in TS._t[:-1]]

