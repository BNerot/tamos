
from tamos.data_IO.data_IO import NamesFormatter
from tamos.element import FuelVector, ElectricityVector
from .network_base import Network


class NonThermalNetwork(Network):

    def __init__(self,
                 hubs_locations,
                 element,
                 properties,
                 production_hub,
                 scale_factor=1,
                 eco_count=True,
                 name=None):
        """
        NonThermalNetwork instances makes possible to share ElectricityVector or FuelVector elements between hubs.

        Power is exchanged between two hubs, given a distribution loss related to the "Losses (%/km)" property.
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

        element : ElectricityVector, FuelVector
            Element exchanged between `hubs`.
        properties : dict {str: int | float}
            Techno-economic properties of the network.
            The `properties` attribute must include the following keys:

            * "Losses (%/km)"
            * "CAPEX (EUR/km)"
            * "OPEX (%CAPEX)"

        production_hub : Hub
            The hub that bears all the distribution losses of the network.
            Must be one of `hubs` and must be able to send `element` to the network.
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

        3. The connection of production hub to every other hub using the network is not mandatory,
           i.e. nothing constrains the network graph to be connected.

        4. The way distribution losses are declared and applied to `production_hub` in the MILP formalism gives no upper bound for
           these losses. This implementation is chosen to prevent the use of the absolute value of a continuous variable,
           which comes with a binary variable.
           The consequence of this implementation is that problem where large losses benefit to the minimization of
           the objective function (unlikely to happen) might get a solution with no physical meaning.

        """
        if name is None:
            name = f"NonThermalNetwork..{element}"
        super().__init__(hubs_locations,
                         properties,
                         production_hub,
                         scale_factor,
                         eco_count,
                         name)
        self._add_used_element(element, "element", (ElectricityVector, FuelVector))


    @property
    def production_hub(self):
        """
        The hub that bears all the distribution losses of the network.
        Must be one of `hubs` and must be able to send `element` to the network.

        """
        return self._production_hub

    @production_hub.setter
    def production_hub(self, production_hub):
        assert production_hub in self._hubs, f"{self}: 'production_hub' must be a used hub."
        self._production_hub = production_hub

    def _declare_variables(self, model_data, TS):
        super()._declare_variables(model_data, TS)
        model_data.add_variable_use("F_N_ABS")
        for hub in self._hubs:
            for hub_2 in self._hubs:
                if id(hub_2) < id(hub):  # does not depend on hub order, i.e. can be declared only once
                    model_data.vars["F_N_ABS"].update(
                                model_data.mdl.continuous_var_dict(
                                                    ((hub, hub_2, self, t) for t in TS._t[:-1]),
                                                    lb=- model_data.inf,
                                                    name=NamesFormatter.fmt("F_N_ABS", [hub], [hub_2], [self], TS._t[:-1])
                                                                  ))


    def _declare_constraints(self, model_data, TS):
        assert self.might_connect(self.production_hub), f"{self}: 'production_hub' must be able to connect to the network"
        super()._declare_constraints(model_data, TS)

        # definition of F_N_ABS
        for hub in self._hubs:
            for hub_2 in self._hubs:
                if id(hub_2) < id(hub):         # no calls to get_F_N to save the time of function calls
                    model_data.cts[self] += model_data.mdl.add_constraints(
                        (model_data.vars["F_N_ABS"][(hub, hub_2, self, TS._t[ind_t])]
                         >= model_data.vars["F_N"][(hub, hub_2, self, TS._t[ind_t])]
                         for ind_t in range(len(TS._t[:-1]))),
                        names=NamesFormatter.fmt("Abs energy transfer 1", [hub], [hub_2], [self], TS._t[:-1])
                                                                          )
                    model_data.cts[self] += model_data.mdl.add_constraints(
                        (model_data.vars["F_N_ABS"][(hub, hub_2, self, TS._t[ind_t])]
                         >= - model_data.vars["F_N"][(hub, hub_2, self, TS._t[ind_t])]
                         for ind_t in range(len(TS._t[:-1]))),
                        names=NamesFormatter.fmt("Abs energy transfer 2", [hub], [hub_2], [self], TS._t[:-1])
                                                                          )
        losses_ = self._get_property("Losses (%/km)") / 100
        losses = [losses_ \
                * model_data.mdl.sum(model_data.vars["F_N_ABS"][(hub, hub_2, self, TS._t[ind_t])]
                                   * self.get_distance(hub, hub_2)
                                     for hub in self._hubs
                                     for hub_2 in self._hubs
                                     if self._connection_can_exist(hub, hub_2) and id(hub_2) < id(hub)
                                    )
                  for ind_t in range(len(TS._t[:-1]))]

        for hub in self._hubs:
            if self.production_hub == hub:
                model_data.cts[self] += model_data.mdl.add_constraints(
                    (model_data.vars["F_SYS"][(hub, self, TS._t[ind_t])]
                     == model_data.mdl.sum(self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                                           for hub_2 in self._hubs
                                           if self._connection_can_exist(hub, hub_2)
                                           )
                     + losses[ind_t]
                     for ind_t in range(len(TS._t[:-1]))),
                    names=NamesFormatter.fmt("Hub multi connections", [hub], [self], TS._t[:-1]))
            else:
                model_data.cts[self] += model_data.mdl.add_constraints(
                    (model_data.vars["F_SYS"][(hub, self, TS._t[ind_t])]
                     == model_data.mdl.sum(self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                                           for hub_2 in self._hubs
                                           if self._connection_can_exist(hub, hub_2)
                                           )
                     for ind_t in range(len(TS._t[:-1]))),
                    names=NamesFormatter.fmt("Hub multi connections", [hub], [self], TS._t[:-1]))



    def _declare_KPIs(self, model_data, TS):
        if self.eco_count:
            total_network_length = model_data.mdl.sum(self.get_distance(hub_1, hub_2) *
                                                      model_data.vars["Y_N"][(hub_1, hub_2, self)]
                                                      for hub_1 in self._hubs
                                                      for hub_2 in self._hubs
                                                      if
                                                      self._connection_can_exist(hub_1, hub_2)
                                                      and (id(hub_2) < id(hub_1)))

            CAPEX = self._get_property("CAPEX (EUR/km)") * total_network_length

            fixed_OPEX = (self._get_property("OPEX (%CAPEX)") / 100) * CAPEX

            cost_network, CAPEX__, OPEX__ = self.compute_actualized_cost(CAPEX,
                                                                         fixed_OPEX,
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