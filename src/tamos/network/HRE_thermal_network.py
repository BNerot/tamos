from numpy import log, vectorize, sqrt, min

from ..data_IO.data_IO import NamesFormatter, DataAccessors
from .network_base import Network
from .thermal_network import ThermalNetwork


class HREThermalNetwork(ThermalNetwork):
    def __init__(self,
                 hubs_locations,
                 element,
                 properties,
                 production_hub,
                 production_mode="heat&cold",
                 scale_factor=1,
                 eco_count=True,
                 name=None):
        """
        HREThermalNetwork is a thermal network model that differs from `ThermalNetwork` by an investment cost
        being dependant from the annual network linear energy density. It is an adaptation of the model described in
        (Persson et al., 2019) ([1]_, [2]_).


        Power is exchanged between two hubs, given a distribution losses proportional to the difference between
        network temperature and soil temperature.
        All distribution losses must be compensated for by an additional power in `production_hub`.


        NonThermalNetwork components are associated with the following exported decision variables:

        * X_N(hub_1, hub_2), binary.
          Whether a connection from hub `hub_1` to hub `hub_2` exists and allows a flow of `element`.
          Note that X_N(hub_1, hub_2) is different from X_N(hub_2, hub_1).
        * Y_N(hub_1, hub_2), binary.
          Whether a connection between hubs `hub_1` and `hub_2` exists and allows a flow of `element`.
        * X_SYS(hub), binary.
          Whether the hub `hub` is connected to at least one other hub, no matter the direction of the connection.
        * For all t, F_SYS(hub, t), continuous, in kW.
          The power related to `element` going from hub `hub` to the network.
        * For all t, F_N(hub_1, hub_2, t), continuous, in kW.
          The power related to `element` going from hub `hub_1` to hub `hub_2` through the network.
          Note that F_N(hub_1, hub_2, t) is the opposite of F_N(hub_2, hub_1, t).

        NonThermalNetwork components declare the following KPIs:

        * `COST_network`
          In euros.
          Contributes to the "Eco" objective function.

        Parameters
        ----------
        hubs_locations : dict {Hub: (float, float)}

            * Keys of `hubs_locations` are the hubs possibly connected by the network.
              They define the `hubs` attribute.
            * Values of `hubs_locations` define x and y coordinates in space.
              In km.
              They describe the position of the hub given the absolute reference (0, 0).
              Used to calculate distance between two hubs. The used distance function can be accessed by the
              `get_distance_function` function and set using `set_distance_function` from tamos.network.

        element : ThermalVectorPair
            Element exchanged between `hubs`.
            Whether `element` is cooled down or warmed up does not define if the network is a heating or cooling network.
        properties : dict {str: int | float}
            Techno-economic properties of the network.
            The `properties` attribute must include the following keys:

            * "Losses (%/km)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"
            * "Used network length (m)"
            * "Capex subsidy (%)"

        production_hub : Hub
            The hub that bears:

             * all the distribution losses of the network.
             * the costs associated with the "Variable OPEX (EUR/MWh)" property.
             * the investment cost, which is associated with the properties
               "OPEX (%CAPEX)", "Used network length (m)" and "Capex subsidy (%)"

            Must be one of `hubs` and must be able to exchange `element` with the network.
        production_mode: {"heat&cold", "heat", "cold"}, optional, default "heat&cold"
            Related to `production_hub`.
            Used to speed up the KPI declaration regarding the `Variable OPEX (EUR/MWh)` property.
            Provides insight on the sign of the flow going from `production_hub` to the network.
            Whether `element` is cooled down or warmed up is independent from `production_mode`.

            * "heat" ("cold"): only heat (cold) is send to the network by `production_hub`.
              The flow is always of the same sign, thus the KPI constraint is like:
              energy = sum(a_given_sign * power(t) * dt)
            * "heat" ("cold"): only heat (cold) is send to the network by `production_hub`.
              The sign of the flow may change during the operation period, thus the KPI constraint is like:
              energy = sum(abs(power(t)) * dt)

            Specifying "heat" or "cold" will speed up the KPI declaration but describes a particular state of energy flows.
            Specifying "heat&cold" makes the KPI declaration long (but has no impact on resolution) but works in all cases.
        scale_factor : float, optional, default 1
            Multiplies the two coordinates of each hub in `hubs_locations`.
            A scale factor >1 tends to increase the distance between two hubs.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.
        name : str, optional

        Notes
        -----
        1. A network component describe an oriented graph where each node is a hub and each edge is a connection between two hubs.

        2. Connection status between two hubs can be defined using the `set_connection_status` and `set_node_status` methods.
           Per default, all pairs of hubs are connected according to the status `no_connection`.

        3. The connection of `production_hub` to every other hub using the network is not mandatory,
           i.e. nothing constrains the network graph to be connected.

        4. The variable OPEX applies only on the thermal production of `production_hub`.
           Another MILP implementation could have taken into account all hubs sending power to the network, at the cost of
           additional continuous decision variables and constraints.

        5. The linear heat density calculation implies a non-linear expression involving the network length L.
           This model relies on the network length being a parameter, "Used network length (m)".

        6. The property "Capex subsidy (%)" decreases the investment cost but does not impact the variable cost.

        References
        ----------
        .. [1] Persson U, Wiechers E, Möller B, Werner S. Heat Roadmap Europe: Heat distribution costs.
           Energy 2019;176:604–22. https://doi.org/10.1016/j.energy.2019.03.189.

        .. [2] Persson U, Werner S. Heat distribution and the future competitiveness of district heating.
           Applied Energy 2011;88:568–76. https://doi.org/10.1016/j.apenergy.2010.09.020.

        """
        super().__init__(hubs_locations=hubs_locations,
                         element=element,
                         properties=properties,
                         production_hub=production_hub,
                         production_mode=production_mode,
                         scale_factor=scale_factor,
                         eco_count=eco_count,
                         name=name)

        TV_IN, TV_OUT = self._element.get_vectors()
        min_DT = min(abs(TV_IN.temperature - TV_OUT.temperature))
        assert min_DT > 0, f"{self}: temperature gap of {self._element} must never be 0."

    def _declare_variables(self, model_data, TS):
        super()._declare_variables(model_data, TS)
        model_data.add_variable_use("CAPEX")
        model_data.vars["CAPEX"][(self,)] = model_data.mdl.continuous_var(
            name=NamesFormatter.fmt_light("CAPEX", self))  # before subsidy is applied

    def _declare_constraints(self, model_data, TS):
        # skips the first parent to redefine network length
        # (from parameters instead of variables like it is the case in ThermalNetwork)
        Network._declare_constraints(self, model_data, TS)
        for hub in self._hubs:
            if self.production_hub == hub:
                used_network_length = self._get_property("Used network length (m)") * 1e-3 * \
                                      model_data.vars["X_SYS"][(self.production_hub, self)]
                model_data.cts[self] += \
                    model_data.mdl.add_constraints((model_data.vars["F_SYS"][(hub, self, TS._t[ind_t])]
                                                    == model_data.mdl.sum(
                                                    self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                                                    for hub_2 in self._hubs
                                                    if (self._connection_can_exist(hub, hub_2))
                                                    )
                                                    + DataAccessors.get2(self._losses, TS._t[ind_t], TS._t[ind_t + 1])
                                                    * used_network_length
                                                    for ind_t in range(len(TS._t[:-1]))
                                                    if DataAccessors.get2(self._element._CpDT, TS._t[ind_t],
                                                                          TS._t[ind_t + 1]) != 0),
                                                   names=NamesFormatter.fmt("Hub multi connections", [hub], [self],
                                                                            TS._t[:-1]))
                                                   # when CpDT=0, production can't send on grid but losses
                                                   # must be compensated for, impossible
            else:
                model_data.cts[self] += model_data.mdl.add_constraints((model_data.vars["F_SYS"][(hub, self, t)]
                                                                        == model_data.mdl.sum(
                    self._get_F_N(model_data, hub, hub_2, t)
                    for hub_2 in self._hubs
                    if self._connection_can_exist(hub, hub_2)

                )
                                                                        for t in TS._t[:-1]),
                                                                       names=NamesFormatter.fmt(
                                                                           "Hub multi connections", [hub], [self],
                                                                           TS._t[:-1]))

    def _declare_KPIs(self, model_data, TS):
        if self.eco_count:
            """
            Capex model from (Persson et al., 2019)
            Persson U, Wiechers E, Möller B, Werner S. Heat Roadmap Europe: Heat distribution costs. Energy 2019;176:604–22.
            https://doi.org/10.1016/j.energy.2019.03.189.
            WARNING: an error in the paper, see (Persson et al. 2011) for accurate definition of average diameter
            Persson U, Werner S. Heat distribution and the future competitiveness of district heating.
            Applied Energy 2011;88:568–76. https://doi.org/10.1016/j.apenergy.2010.09.020.
            """
            variable_OPEX = self._get_property("Variable OPEX (EUR/MWh)")
            used_network_length = self._get_property("Used network length (m)")
            CAPEX_subsidy = self._get_property("Capex subsidy (%)")

            if self.production_mode != "heat&cold":     # quicker declaration (no abs variables)
                sign1 = 1 if self.element.is_cooled else -1
                sign2 = 1 if self.production_mode == "heat" else -1
                E_ABS_OPEX = sign1 * sign2 * model_data.mdl.sum(
                                                                model_data.vars["F_SYS"][(self.production_hub, self, TS._t[ind_t])]
                                                                * TS._dt[ind_t]
                                                                   for ind_t in range(len(TS._t[:-1]))) * TS._step_value
            else:                                       # birdirectionnal compatibility
                E_ABS_OPEX = model_data.mdl.sum(
                    model_data.mdl.abs(model_data.vars["F_SYS"][(self.production_hub, self, TS._t[ind_t])])
                                                                   * TS._dt[ind_t]
                                                                   for ind_t in range(len(TS._t[:-1]))) * TS._step_value
            variable_OPEX = (variable_OPEX / 1e3) * E_ABS_OPEX

            network_linear_density = E_ABS_OPEX * ((3600 / 1e6) / used_network_length)  # in GJ/m
            # piecewise linear function of the log in average diameter definition
            pwlf = model_data.create_pwlf(log, vectorize(lambda x: 1 / x, otypes=["float64"]), inf=1.5, sup=15,
                                          nbr_points=1)
            # average diameter depends on the network linear heat density
            d_1 = 0.0486 * pwlf(network_linear_density) + 0.0007
            d_2 = 0.02
            mean_diameter = model_data.mdl.max(d_1, d_2)

            TV_IN, TV_OUT = self._element.get_vectors()
            min_DT = min(abs(TV_IN.temperature - TV_OUT.temperature))
            model_data.cts[self] += [model_data.mdl.add_indicator(model_data.vars["X_SYS"][(self.production_hub, self)],
                                                                  model_data.vars["CAPEX"][(self,)] == used_network_length
                                                                  * (212 + 4464 * sqrt(50 / min_DT)
                                                                     * mean_diameter),
                                                                  name=NamesFormatter.fmt_light("CAPEX ON", self))]
            model_data.cts[self] += [model_data.mdl.add_indicator(model_data.vars["X_SYS"][(self.production_hub, self)],
                                                                  model_data.vars["CAPEX"][(self,)] == 0,
                                                                  active_value=0,
                                                                  name=NamesFormatter.fmt_light("CAPEX OFF", self))]
            fixed_OPEX = (self._get_property("OPEX (%CAPEX)") / 100) * model_data.vars["CAPEX"][(self,)]

            CAPEX = (1 - CAPEX_subsidy / 100) * model_data.vars["CAPEX"][(self,)]

            cost_network, CAPEX__, OPEX__ = self.compute_actualized_cost(CAPEX,
                                                                         fixed_OPEX + variable_OPEX,
                                                                         TS._system_lifetime)

            base_name = "COST_network"
            for name_, value, used_in_solving in [(f"{base_name} - CAPEX", CAPEX__, False),
                                                  (f"{base_name} - OPEX", OPEX__, False),
                                                  (f"{base_name}", cost_network, True),
                                                  ]:
                model_data.add_KPI_use(name_, ("Network",))
                name = NamesFormatter.fmt_light(name_, self, KPI=True)
                kpi = model_data.mdl.add_kpi(value, publish_name=name)
                model_data.KPIs[self] += [model_data.KPI_wrapper(kpi, "Eco", used_in_solving)]
