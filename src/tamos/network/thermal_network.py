from numpy import minimum, maximum

from ..data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair
from .network_base import Network


class ThermalNetwork(Network):
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
        ThermalNetwork instances makes possible to share ThermalVectorPair elements between hubs.

        Power is exchanged between two hubs, given a distribution losses proportional to the difference between
        network temperature and soil temperature.
        All distribution losses must be compensated for by an additional power in `production_hub`.
        The investment cost is proportional to the network length.


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
            * "CAPEX (EUR/km)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        production_hub : Hub
            The hub that bears:

             * all the distribution losses of the network.
             * the costs associated with the "Variable OPEX (EUR/MWh)" property.

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

        """
        if name is None:
            name = f"ThermalNetwork..{element}"
        super().__init__(hubs_locations,
                         properties,
                         production_hub,
                         scale_factor,
                         eco_count,
                         name)
        self._add_used_element(element, "element", _ThermalVectorPair)
        self.production_mode = production_mode
        self.set_soil_properties(273+10, 0.7)
        print(f"{self}: Soil properties set with default values.")



    @property
    def production_hub(self):
        """
        The hub that bears:

         * all the distribution losses of the network.
         * the costs associated with the "Variable OPEX (EUR/MWh)" property.

        Must be one of `hubs` and must be able to exchange `element` with the network.

        """
        return self._production_hub

    @production_hub.setter
    def production_hub(self, production_hub):
        assert production_hub in self._hubs, f"{self}: 'production_hub' must be a used hub."
        self._production_hub = production_hub


    @property
    def production_mode(self):
        """
        Related to `production_hub`.

        {"heat&cold", "heat", "cold"}, optional, default "heat&cold"
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


        """
        return self._production_mode

    @production_mode.setter
    def production_mode(self, production_mode):
        assert production_mode in ["heat", "cold", "heat&cold"], f"{self}: 'production_mode' must be one of 'heat', 'cold', 'heat&cold'."
        self._production_mode = production_mode

    def set_soil_properties(self, soil_temperature, U,
                            losses_direction: "'Both', 'Heat losses', 'Heat gains'" = "Both"):
        """
        Sets the physical properties that define thermal losses.

        Parameters
        ----------
        soil_temperature : int, float or numpy.ndarray
            In Kelvins (K).
            The temperature of the soil at the buried depth of the network pipes.
        U : int, float or numpy.ndarray
            In W/(m.K).
            Specific heat loss per routed meter. Includes both supply  and return pipes.
        losses_direction : {'Both', 'Heat losses', 'Heat gains'}, optional, default 'Both'
            Direction of the heat exchanges between the network infrastructure and the soil.
            Specifying 'Heat losses' ('Heat gains') allows to prevent from happening the case where
            cold (heat) must be produced in `production_hub` to compensate thermal gains (losses) in
            a district heating (district cooling) network, when thermal demand is lower than soil thermal exchanges.

             * 'Heat losses': only thermal energy exchanges from the network to the soil are taken into account,
               others are set to 0.
             * 'Heat gains': only thermal energy exchanges from the soil to the network are taken into account,
               others are set to 0.
             * 'Both': all thermal energy exchanges are taken into account.

        Notes
        -----
        1. The thermal energy exchanged between the network infrastructure and the soil is like:
           thermal_exchanges(t) = sign * U * network_length * ((T_warm(t)+T_cold(t)) / 2 - T_soil(t))
           With:

            * network_length: the length of all the edges in the network.
              If connections are bidirectional, length is accounted for only once.
            * T_warm(t): temperature of the warm vector of `element`
            * T_cold(t): temperature of the cold vector of `element`
            * T_soil(t): temperature of the soil

        2. By default, set_soil_properties is called with `soil_temperature` =273+10, `U` =0.7, `losses_direction` ='Both'.

        3. This method does not assume pipes are laid down underground: setting a `U` value according to the `soil_temperature`
           value is enough to describe any surrounding environment of the pipes.

        """
        DataAccessors.type_checker(soil_temperature, self, "soil_temperature", "numeric")
        DataAccessors.type_checker(U, self, "U", "numeric")
        if losses_direction not in ['Both', 'Heat losses', 'Heat gains']:
            raise ValueError(f"{self}: Incorrect value for 'losses_direction'.")
        self._soil_temperature = soil_temperature
        self._losses_direction = losses_direction
        self._U = U
        self._compute_heat_losses()

    def _declare_constraints(self, model_data, TS):
        assert self.might_connect(self.production_hub), f"{self}: 'production_hub' must be able to connect to the network"
        super()._declare_constraints(model_data, TS)
        for hub in self._hubs:
            if self.production_hub == hub:
                total_network_length = model_data.mdl.sum(self.get_distance(hub_1, hub_2) *
                                                          model_data.vars["Y_N"][(hub_1, hub_2, self)]
                                                          for hub_1 in self._hubs
                                                          for hub_2 in self._hubs
                                                          if
                                                          (self._connection_can_exist(hub_1,
                                                                                      hub_2))
                                                          and (id(hub_2) < id(hub_1)))


                model_data.cts[self] += model_data.mdl.add_constraints(
                    (model_data.vars["F_SYS"][(hub, self, TS._t[ind_t])]
                     == model_data.mdl.sum(self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                                                for hub_2 in self._hubs
                                                if self._connection_can_exist(hub, hub_2)
                                                )
                     + DataAccessors.get2(self._losses, TS._t[ind_t], TS._t[ind_t + 1])
                     * total_network_length
                     for ind_t in range(len(TS._t[:-1]))
                     if DataAccessors.get2(self._element._CpDT,
                                           TS._t[ind_t],
                                           TS._t[ind_t + 1]) != 0),
                    # when CpDT=0, production can't send on grid but losses must be compensated for, impossible
                    names=NamesFormatter.fmt("Hub multi connections", [hub], [self], TS._t[:-1]))
            else:
                model_data.cts[self] += model_data.mdl.add_constraints((model_data.vars["F_SYS"][(hub, self, t)]
                                == model_data.mdl.sum(self._get_F_N(model_data, hub, hub_2, t)
                                                for hub_2 in self._hubs
                                                if self._connection_can_exist(hub, hub_2)
                                                )
                                                                        for t in TS._t[:-1]),
                                                                       names=NamesFormatter.fmt(
                                                                           "Hub multi connections", [hub], [self],
                                                                           TS._t[:-1]))


    def _compute_heat_losses(self):
        T_soil = self._soil_temperature
        TV_IN, TV_OUT = self._element.get_vectors()
        mean_temperature = (TV_IN.temperature + TV_OUT.temperature) / 2  # first "mean": supply and return
        sign_ = -1 if self._element.is_cooled else 1
        heat_gains = - self._U * (mean_temperature - T_soil)
        if self._losses_direction == "Both":
            heat_gains = heat_gains
        elif self._losses_direction == "Heat losses":
            heat_gains = minimum(heat_gains, 0)
        else:
            heat_gains = maximum(heat_gains, 0)
        self._losses = heat_gains * sign_


    def _declare_KPIs(self, model_data, TS):
        if self.eco_count:
            variable_OPEX = self._get_property("Variable OPEX (EUR/MWh)")
            total_network_length = model_data.mdl.sum(self.get_distance(hub_1, hub_2) *
                                                      model_data.vars["Y_N"][(hub_1, hub_2, self)]
                                                      for hub_1 in self._hubs
                                                      for hub_2 in self._hubs
                                                      if
                                                      self._connection_can_exist(hub_1, hub_2)
                                                      and (id(hub_2) < id(hub_1)))
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


            variable_cost = (variable_OPEX / 1000) * E_ABS_OPEX

            CAPEX = self._get_property("CAPEX (EUR/km)") * total_network_length
            fixed_OPEX = (self._get_property("OPEX (%CAPEX)") / 100) * CAPEX

            cost_network, CAPEX__, OPEX__ = self.compute_actualized_cost(CAPEX,
                                                                         fixed_OPEX + variable_cost,
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