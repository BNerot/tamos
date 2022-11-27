from numpy import array, maximum, mean

from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair
from .production_base import Production


class FPSolar(Production):

    def __init__(self,
                 energy_sink,
                 air_temperature,
                 total_irradiance,
                 properties,
                 pinch = 3,
                 given_sizing=None, name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True
                 ):
        """
        FPSolar components convert solar irradiance to heat.

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
        energy_sink: ThermalVectorPair
            Thermal flow that is warmed up.
        air_temperature: int, float or numpy.ndarray
            In Kelvins (K).
            Temperature of the air surrounding the solar panels.
        total_irradiance: int, float or numpy.ndarray
            In kW/m².
            Solar irradiance received on the normal of the solar panels.
        properties: dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/m2)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"
            * "eta0"
            * "a1 (W/(m2.K))"
            * "a2 (W/(m2.K2))"
            * "a5 (J/(m2.K))"
            * "UB area (m2)"

        pinch : int, float or numpy.ndarray, optional, default 3
            Difference between the temperature of the fluid circulating in the solar panels and the one of `energy_sink`.
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


        Notes
        -----
        1. Parameters "eta0", "a1 (W/(m2.K))", "a2 (W/(m2.K2))" and "a5 (J/(m2.K))" can be found in databases like the
           Solar Keymark database.

        2. 'FPSolar' stands for 'Flat plate solar thermal'.

        """
        if name is None:
            name = f"STh..{energy_sink}"
        super().__init__(properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self._add_used_element(energy_sink, "energy_sink", _ThermalVectorPair)
        assert energy_sink.is_warmed, f"{self}: '{energy_sink}' must be warmed up. consider using ~({energy_sink})."

        self.pinch = pinch
        self.total_irradiance = total_irradiance
        self.air_temperature = air_temperature

    @property
    def energy_sink(self):
        """
        Thermal flow that is warmed up.
        ThermalVectorPair

        """
        return self._energy_sink

    @property
    def pinch(self):
        """
        Difference between the temperature of the fluid circulating in the solar panels and the one of `energy_sink`.
        int, float or numpy.ndarray

        """
        return self._pinch

    @pinch.setter
    def pinch(self, pinch):
        DataAccessors.type_checker(pinch, self, "pinch", "numeric")
        self._pinch = pinch

    @property
    def total_irradiance(self):
        """
        Solar irradiance received on the normal of the solar panels.
        In kW/m².
        int, float or numpy.ndarray

        """
        return self._total_irradiance

    @total_irradiance.setter
    def total_irradiance(self, total_irradiance):
        DataAccessors.type_checker(total_irradiance, self, "total_irradiance", "numeric")
        self._total_irradiance = total_irradiance

    @property
    def air_temperature(self):
        """
        Temperature of the air surrounding the solar panels.
        In Kelvins (K).
        int, float or numpy.ndarray

        """
        return self._air_temperature

    @air_temperature.setter
    def air_temperature(self, air_temperature):
        DataAccessors.type_checker(air_temperature, self, "air_temperature", "numeric")
        self._air_temperature = air_temperature

    def _declare_variables(self, model_data, hub, TS):
        super()._declare_variables(model_data, hub, TS)
        model_data.add_variable_use("STh_Area")
        model_data.vars["STh_Area"][(hub, self)] = model_data.mdl.continuous_var(
            name=NamesFormatter.fmt_light("STh_Area", hub, self))

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        eta0 = self._get_property("eta0")
        a1 = self._get_property("a1 (W/(m2.K))") / 1e3
        a2 = self._get_property("a2 (W/(m2.K2))") / 1e3
        a5 = self._get_property("a5 (J/(m2.K))") / 3.6e6
        UB_area = self._get_property("UB area (m2)")

        TV_IN, TV_OUT = self.energy_sink.get_vectors()
        mean_temperature = (TV_IN.temperature + TV_OUT.temperature) / 2 + self.pinch

        def get_losses(T, T_air):
            DT = T - T_air
            return DT * (a1 + a2 * DT)

        F_losses = get_losses(mean_temperature, self.air_temperature)                   # in kW/m²

        F_inertia = a5 * mean(mean_temperature - self.air_temperature) / TS._step_value  # in kW/m²
        production_serie = eta0 * self.total_irradiance - F_losses - F_inertia
        production_serie = array([DataAccessors.get2(production_serie, TS._t[ind_t], TS._t[ind_t + 1])
                                  for ind_t in range(len(TS._t[:-1]))])
        production_serie = maximum(production_serie, 0)
        self._annual_max_areal_power = max(production_serie)

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_P"][(hub, self._energy_sink, self, TS._t[ind_t])]
             == production_serie[ind_t] * model_data.vars["STh_Area"][(hub, self)]
             for ind_t in range(len(TS._t[:-1]))),
            names=NamesFormatter.fmt("Power balance", [hub], [self], TS._t[:-1]))

        model_data.cts[hub][self] += [model_data.mdl.add_constraint(
            model_data.vars["SP_P"][(hub, self)]
            == self._annual_max_areal_power * model_data.vars["STh_Area"][(hub, self)],
            ctname=NamesFormatter.fmt_light("Sizing area", hub, self))]

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["Q_P"][(hub, self, t)]
             == model_data.vars["F_P"][(hub, self._energy_sink, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Ref production", [hub], [self], TS._t[:-1]))

        model_data.cts[hub][self] += [
            model_data.mdl.add_constraint((model_data.vars["STh_Area"][(hub, self)] <= UB_area),
                                          ctname=NamesFormatter.fmt_light("Area UB", hub, self))]

        [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_lb(0) for t in TS._t[:-1]]

    def _declare_KPIs(self, model_data, hub, TS):
        # method overload because the given CAPEX is per area of collectors
        if self.eco_count:
            CAPEX = model_data.vars["STh_Area"][(hub, self)] * self._get_property("CAPEX (EUR/m2)")

            fixed_OPEX = (self._get_property("OPEX (%CAPEX)") / 100) * CAPEX

            annual_production = model_data.mdl.sum(model_data.vars["Q_P"][(hub, self, TS._t[ind_t])]
                                                   * TS._dt[ind_t]
                                                   for ind_t in range(len(TS._t[:-1])))  * TS._step_value
            variable_OPEX = (annual_production / 1e3) * self._get_property("Variable OPEX (EUR/MWh)")

            cost_production, CAPEX__, OPEX__ = self.compute_actualized_cost(CAPEX,
                                                                            fixed_OPEX + variable_OPEX,
                                                                            TS._system_lifetime)

            base_name = "COST_production"
            for name_, value, used_in_solving in [(f"{base_name} - CAPEX", CAPEX__, False),
                                                  (f"{base_name} - OPEX", OPEX__, False),
                                                  (f"{base_name}", cost_production, True),
                                                  ]:
                model_data.add_KPI_use(name_, ("Hub", "Production"))
                name = NamesFormatter.fmt_light(name_, hub, self, KPI=True)
                kpi = model_data.mdl.add_kpi(value, publish_name=name)
                model_data.KPIs[hub][self] += [model_data.KPI_wrapper(kpi, "Eco", used_in_solving)]
