from matplotlib.pyplot import subplots
from networkx import set_node_attributes, set_edge_attributes, Graph, DiGraph, minimum_spanning_tree, get_node_attributes, draw_networkx
from numpy import array, all
from scipy.spatial.distance import cityblock

from tamos.component import Component
from ..data_IO.data_IO import DataAccessors, NamesFormatter


class Status(tuple):
    def __repr__(self):
        return self[0]

class Network(Component):
    no_connection = Status(("No connection", None))
    connection = Status(("Connection", "Black"))
    optim_two_ways = Status(("Optim two ways", "Green"))
    optim_one_way_min = Status(("Optim one way min", "Orange"))
    optim_one_way_max = Status(("Optim one way max", "Red"))


    _distance_function = cityblock

    def __init__(self,
                 hubs_locations,
                 properties,
                 production_hub,
                 scale_factor,
                 eco_count,
                 name):

        super().__init__(name, properties)
        self.scale_factor = scale_factor
        self._set_hubs_locations(hubs_locations)
        self.production_hub = production_hub
        self.eco_count = eco_count

    @staticmethod
    def _get_distance_function():
        """
        Returns the function that gives the distance in km between two hubs.

        """
        return Network._distance_function

    @staticmethod
    def _set_distance_function(distance_function):
        """
        Defines the function that gives the distance in km between two hubs.

        Parameters
        ----------
        distance_function : callable, f(u, v), that meets the following definition:

        * `u` is a list [x, y] where `x` and `y` are space coordinates of point `u` (same for `v`).
        * f(u, v) = f(v, u)
        * f(u, v) returns a distance in km

        Notes
        -----
        Default distance function is `scipy.spatial.distance.cityblock`.

        """

        message = f"Invalid 'distance_function'."
        try:
            res1 = distance_function([0, 1], [2, 3])
            res2 = distance_function([2, 3], [0, 1])
        except Exception:
            raise ValueError(message)
        assert isinstance(res1, (int, float)), message
        # weak assert below
        assert res1 == res2, f"'distance_function' must be symmetric: " \
                             f"distance_function(h_1, h_2) = distance_function(h_2, h_1)."
        print(f"Warning: The change in 'distance_function' has no effect on already defined networks."
              f" It will impact all future networks.")
        Network._distance_function = distance_function

    @property
    def scale_factor(self):
        """
        Multiplies the two coordinates of each hub in `hubs_locations`.

        A scale factor >1 tends to increase the distance between two hubs.
        float

        """
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, scale_factor):
        if scale_factor is not None:
            DataAccessors.type_checker(scale_factor, self, "scale_factor", (int, float))
        self._scale_factor = scale_factor

    @property
    def eco_count(self):
        """
        Whether this instance contributes to the system "Eco" KPI
        bool

        """
        return self._eco_count

    @eco_count.setter
    def eco_count(self, eco_count):
        DataAccessors.type_checker(eco_count, self, "eco_count", bool)
        self._eco_count = eco_count

    @property
    def element(self):
        """
        Element exchanged between `hubs`.

        Whether `element` is cooled down or warmed up does not define if the network is a heating or cooling network.

        """
        return self._element

    @property
    def hubs(self):
        """
        The hubs involved in the network definition.

        Some of these hubs might be completely disconnected from the network
        (i.e. network.might_connect(hub) is False) yet they are still considered and associated with MILP decision variables.

        """
        return self._hubs


    def _set_hubs_locations(self, hubs_locations):
        self._hubs = set(hubs_locations)
        self._create_location_graph(hubs_locations)
        self._create_network_graph()

    def _create_location_graph(self, hubs_locations):
        coordinates = {hub: array(coordinates) * self.scale_factor
                       for hub, coordinates in hubs_locations.items()}
        self._location_graph = Graph()
        self._location_graph.add_nodes_from(self._hubs)
        set_node_attributes(self._location_graph, coordinates, 'coord')
        for hub_1 in self._hubs:
            for hub_2 in self._hubs:
                if hub_1 != hub_2:
                    distance = self.get_distance(hub_1, hub_2)
                    self._location_graph.add_edge(hub_1, hub_2, weight=distance)

    def _create_network_graph(self):
        self._network_graph = DiGraph(self._location_graph.copy())
        set_edge_attributes(self._network_graph, self.no_connection, "status")

    def generate_MSP(self, excluded_hubs=None):
        """
        Defines a minimum spanning tree (MSP) linking all hubs of `hubs` that are not in `excluded_hubs`.

        The MSP calculation does not account for bidirectionality thus all edges of the MSP graph are given
        the status `optim_two_ways`.
        The status of the other edges (not part of the MSP) are not modified.

        Parameters
        ----------
        excluded_hubs : list of Hub, optional
            All hubs from `excluded_hubs` must be part of `hubs`.

        """
        if excluded_hubs is None:
            excluded_hubs = []
        else:
            for hub in excluded_hubs:
                assert hub in self._hubs, f"{self}: object {hub} from 'excluded_hubs' is not a hub used by network."
        included_hubs = [hub for hub in self._hubs if hub not in excluded_hubs]
        msp = minimum_spanning_tree(self._network_graph.subgraph(included_hubs).to_undirected(),
                                    weight="weight")
        for edge in msp.edges:
            hub_1, hub_2 = edge
            self.set_connection_status(hub_1, hub_2, Network.optim_two_ways)

    def set_status(self, status, excluded_hubs=None):
        """
        Define the connection status of all edges except the ones involving a hub from `excluded_hubs`.

        Parameters
        ----------
        status : `status` may take 5 values that are attributes of this instance:

            * no_connection: flow of `element` is forbidden

            * connection: flow of `element` is possible

            * optim_one_way_min: flow of `element` must exist in at least one direction
              (from hub_1 to hub_2, from hub_2 to hub_1 or in both directions)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_one_way_max: flow of `element` may exist in at most one direction
              (from hub_1 to hub_2 or from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_two_ways: flow of `element` may exist in both directions
              (from hub_1 to hub_2 and from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

        excluded_hubs : list of Hub, optional
            All hubs from `excluded_hubs` must be part of `hubs`.

        """
        if excluded_hubs is None:
            excluded_hubs = set()
        else:
            for hub in excluded_hubs:
                assert hub in self._hubs, f"{self}: object '{hub}' from 'excluded_hubs' is not a hub used by network."
            excluded_hubs = set(excluded_hubs)
        for hub_1 in self._hubs.difference(excluded_hubs):
            for hub_2 in self._hubs.difference(excluded_hubs):
                if hub_1 != hub_2:
                    self.set_connection_status(hub_1, hub_2, status)

    def _connection_can_exist(self, hub_1, hub_2):
        return (self.get_connection_status(hub_1, hub_2) != self.no_connection) \
           or (self.get_connection_status(hub_2, hub_1) != self.no_connection)


    def get_connection_status(self, hub_1, hub_2):
        """
        Returns the status of the directional connection between two hubs.

        Parameters
        ----------
        hub_1, hub_2 : Hub
            Must be hubs from `hubs`.

        Returns
        -------
        network status
            The status of the edge from hub_1 to hub_2.

        """
        assert (hub_1 in self._hubs) and (hub_2 in self._hubs), \
            f"{self}: objects '{hub_1}' and '{hub_2}' are not hubs used by network."
        if hub_1 == hub_2:
            return self.no_connection
        else:
            return self._network_graph[hub_1][hub_2]["status"]

    def set_connection_status(self, hub_1, hub_2, status):
        """
        Defines the connection status of the edge going from `hub_1` to `hub_2`.

        Parameters
        ----------
        hub_1, hub_2 : Hub
            Must be hubs from `hubs`.
        status : `status` may take 5 values that are attributes of this instance:

            * no_connection: flow of `element` is forbidden

            * connection: flow of `element` is possible

            * optim_one_way_min: flow of `element` must exist in at least one direction
              (from hub_1 to hub_2, from hub_2 to hub_1 or in both directions)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_one_way_max: flow of `element` may exist in at most one direction
              (from hub_1 to hub_2 or from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_two_ways: flow of `element` may exist in both directions
              (from hub_1 to hub_2 and from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

        Notes
        -----
        Given two hubs `hub_1` and `hub_2`, the order of calls to `set_connection_status` matters.

        >>> network.set_connection_status(hub_1, hub_2, network.no_direction)
        >>> network.set_connection_status(hub_2, hub_1, network.optim_one_way_max)
        >>> network.get_connection_status(hub_2, hub_1) == network.get_connection_status(hub_1, hub_2)
            True

        The first line has no effect because the second one defines again the connection from `hub_1` to `hub_2`.

        """
        assert (hub_1 in self._hubs) and (hub_2 in self._hubs), \
            f"{self}: objects '{hub_1}' and '{hub_2}' are not hubs used by network."
        assert status in [self.optim_two_ways, self.optim_one_way_max, self.optim_one_way_min, self.no_connection, self.connection], \
            f"{self}: Unkwown  status '{status}'."

        might_connect = [self.might_connect(hub_1), self.might_connect(hub_2)]
        if (hub_1 == hub_2) and (status != self.no_connection):
            print(f"Warning: {self}: a hub cannot be connected to itself, status is '{self.no_connection}'.")
        self._network_graph[hub_1][hub_2]["status"] = status
        if status in [self.optim_two_ways, self.optim_one_way_max, self.optim_one_way_min]:
            self._network_graph[hub_2][hub_1]["status"] = status
        for hub, might_connect_ in zip([hub_1, hub_2], might_connect):
            if self.might_connect(hub) != might_connect_:
                if might_connect_:
                    hub._change_set_attribute("_possibly_connected_networks", [self], add=False, remove=True)
                else:
                    hub._change_set_attribute("_possibly_connected_networks", [self], add=True, remove=False)

    def get_connection_power_bounds(self, hub_1, hub_2):
        """
        Returns the power limits of the flow of element from hub `hub_1` to hub `hub_2`.

        Parameters
        ----------
        hub_1, hub_2 : Hub
            Must be hubs from `hubs`.

        Returns
        -------
        The lower bound and upper bound of the power flow from `hub_1` to `hub_2`, in kW.

        """
        assert (hub_1 in self._hubs) and (hub_2 in self._hubs), \
            f"{self}: objects '{hub_1}' and '{hub_2}' are not hubs used by network."
        return self._network_graph[hub_1][hub_2].get("Min flow", None), \
               self._network_graph[hub_1][hub_2].get("Max flow", None)

    def set_connection_power_bounds(self, hub_1, hub_2, power_lb=None, power_ub=None):
        """
        Sets power limits on the flow of element from hub `hub_1` to hub `hub_2`.
        These limits apply only if the connection from `hub_1` to `hub_2` is used.

        Parameters
        ----------
        hub_1, hub_2 : Hub
            Must be hubs from `hubs`.
        power_lb, power_ub: int, float or numpy.ndarray
            power_lb >= 0, power_ub >= 0
            In kW.
            The lower bound (upper bound) of the flow of `element` from `hub_1` to `hub_2`.

        Examples
        --------
        >>> network.set_connection_power_bounds(hub_1, hub_2, power_lb=400, power_ub=2000)

        If the connection `hub_1` to `hub_2` exists, the power that flows from `hub_1` to `hub_2` must always be in the range
        [0.4, 2] MW.

        >>> network.set_connection_power_bounds(hub_1, hub_2, power_lb=1000)
        >>> network.set_connection_power_bounds(hub_2, hub_1, power_ub=-1000)

        Both calls perform the same operation, but second call is forbidden to make things clearer.

        >>> network.set_connection_power_bounds(hub_1, hub_2, power_lb=1000)
        >>> network.set_connection_power_bounds(hub_2, hub_1, power_lb=2000)

        The constraints implied by these calls make impossible the existence of both connections
        (from `hub_1` to `hub_2` and `hub_2` to `hub_1`): if 1000 kW of `element` flows from `hub_1` to `hub_2`
        then -1000 k> flows from `hub_2` to `hub_1` (and vice versa).

        """
        assert (hub_1 in self._hubs) and (hub_2 in self._hubs), \
            f"{self}: objects '{hub_1}' and '{hub_2}' are not hubs used by network."
        if hub_1 == hub_2:
            raise ValueError(f"{self}: Cannot connect a hub with itself")
        if power_lb is not None:
            DataAccessors.type_checker(power_lb, self, "power_lb", "numeric")
            assert all(power_lb>=0), f"{self}: 'power_lb' must be positive. " \
                                    f"Same effect can be obtained using 'power_ub'" \
                                    f" in '{self}.set_connection_power_bounds({hub_2}, {hub_1}, ...)'."
            self._network_graph[hub_1][hub_2]["Min flow"] = power_lb
        if power_ub is not None:
            DataAccessors.type_checker(power_ub, self, "power_ub", "numeric")
            assert all(power_ub>=0), f"{self}: 'power_ub' must be positive. " \
                                    f"Same effect can be obtained using 'power_lb'" \
                                    f" in '{self}.set_connection_power_bounds({hub_2}, {hub_1}, ...)'."
            self._network_graph[hub_1][hub_2]["Max flow"] = power_ub

    def set_node_status(self, hub, status):
        """
        Defines the connection status of every incoming and outcoming edge of a hub.

        Parameters
        ----------
        hub : Hub
            Must be from `hubs`.
        status : `status` may take 5 values that are attributes of this instance:

            * no_connection: flow of `element` is forbidden

            * connection: flow of `element` is possible

            * optim_one_way_min: flow of `element` must exist in at least one direction
              (from hub_1 to hub_2, from hub_2 to hub_1 or in both directions)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_one_way_max: flow of `element` may exist in at most one direction
              (from hub_1 to hub_2 or from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

            * optim_two_ways: flow of `element` may exist in both directions
              (from hub_1 to hub_2 and from hub_2 to hub_1)
              This status also defines the opposite status, e.g. defining 'hub_1 to hub_2' status defines the one of 'hub_2 to hub_1'.

        """
        assert (hub in self._hubs), \
            f"{self}: '{hub}' is not a hub used by network."
        assert status in [self.optim_two_ways, self.optim_one_way_max, self.optim_one_way_min, self.no_connection, self.connection], \
            f"{self}: Unkwown  status '{status}'."
        if status in [self.connection, self.no_connection]:
            # status(h1, h2) is different from (h2, h1), see 'set_edge_status'
            for hub_2 in self._hubs.difference({hub}):
                self.set_connection_status(hub, hub_2, status)
                self.set_connection_status(hub_2, hub, status)
        else:
            for hub_2 in self._hubs.difference({hub}):
                self.set_connection_status(hub, hub_2, status)

    def might_connect(self, hub):
        """
        Checks whether a hub is able to connect to the network.

        Parameters
        ----------
        hub : Hub

        Returns
        -------
        True if the following two conditions are met:

        * Hub `hub` is one of `hubs`.
        * There exists at least one hub `hub_2` in `hubs` such that connection status from `hub` to `hub_2` or `hub_2` to `hub`
          is different from 'No connection'.

        """
        if hub not in self._hubs:
            print(f"{self}: object '{hub}' is not a hub used by network.")
            return False
        return any([self.get_connection_status(hub, hub_2) != self.no_connection for hub_2 in self._hubs]
                   + [self.get_connection_status(hub_2, hub) != self.no_connection for hub_2 in self._hubs])

    def get_distance(self, hub_1, hub_2):
        """
        Returns the distance between two hubs.
        The used distance function can be accessed by the `get_distance_function` function
        and set using `set_distance_function` from tamos.network.

        Parameters
        ----------
        hub_1, hub_2 : Hub
            Must be from `hubs`.

        Returns
        -------
        The distance from `hub_1` to `hub_2` which is the same than from `hub_2` to `hub_1`

        """
        assert (hub_1 in self._hubs) and (hub_2 in self._hubs), \
            f"{self}: objects '{hub_1}' and '{hub_2}' are not hubs used by network."
        x_1, y_1 = self._location_graph.nodes[hub_1]["coord"]
        x_2, y_2 = self._location_graph.nodes[hub_2]["coord"]
        return Network._distance_function([x_1, y_1], [x_2, y_2])

    def _declare_variables(self, model_data, TS):
        """
        if element is a ThermalVectorPair, definition of var, const and obj is done on the incoming vector of TVP
        (forward flow)
        and a set of opposite constraints is defined to account for the returning flow
        :return:
        """

        # Y_N(h1, h2) = 1 if and only if two hubs (h1, h2) are connected in at least one direction.
        # Useful for declaring the same investment cost no matter the existence of bidirectionality
        model_data.add_variable_use("F_N", ("Hub", "Hub", "Network", "Date"))
        model_data.add_variable_use("X_N", ("Hub", "Hub", "Network"))
        model_data.add_variable_use("F_SYS", ("Hub", "Network", "Date"))
        model_data.add_variable_use("X_SYS", ("Hub", "Network"))
        model_data.add_variable_use("Y_N", ("Hub", "Hub", "Network"))

        for hub in self._hubs:
            model_data.vars["F_SYS"].update(model_data.mdl.continuous_var_dict(((hub, self, t) for t in TS._t[:-1]),
                                                                               lb=- model_data.inf,
                                                                               name=NamesFormatter.fmt("F_SYS",
                                                                                                       [hub],
                                                                                                       [self],
                                                                                                       TS._t[:-1])))
            model_data.vars["X_SYS"][(hub, self)] = model_data.mdl.binary_var(
                name=NamesFormatter.fmt_light("X_SYS", hub, self))
            for hub_2 in self._hubs:
                if hub != hub_2:
                    model_data.vars["X_N"][(hub, hub_2, self)] = model_data.mdl.binary_var(
                        name=NamesFormatter.fmt_light("X_N", hub, hub_2, self))
                if id(hub_2) < id(hub):  # does not depend on hub order, i.e. can be declared only once
                    model_data.vars["Y_N"][(hub, hub_2, self)] = model_data.mdl.binary_var(
                        name=NamesFormatter.fmt_light("Y_N", hub, hub_2, self))

                    model_data.vars["F_N"].update(
                                model_data.mdl.continuous_var_dict(
                                                    ((hub, hub_2, self, t) for t in TS._t[:-1]),
                                                    lb=- model_data.inf,
                                                    name=NamesFormatter.fmt("F_N", [hub], [hub_2], [self], TS._t[:-1])
                                                                  ))

    def _get_F_N(self, model_data, hub_1, hub_2, t):
        """
        Helper function to deal with F_N(hub_1, hub_2) = - F_N(hub_2, hub_1).
        """
        if id(hub_2) < id(hub_1):
            return model_data.vars["F_N"][(hub_1, hub_2, self, t)]
        else:
            return - model_data.vars["F_N"][(hub_2, hub_1, self, t)]



    def _declare_constraints(self, model_data, TS):
        model_data.cts[self] = []
        for hub in self._hubs:
            connection_vars = []
            for hub_2 in self._hubs:
                if hub != hub_2:
                    min_flow, max_flow = self.get_connection_power_bounds(hub, hub_2)
                    if min_flow is not None:
                        model_data.cts[self] += model_data.mdl.add_indicators(
                            [model_data.vars["X_N"][(hub, hub_2, self)] for t in TS._t[:-1]],
                            [DataAccessors.get2(min_flow, TS._t[ind_t], TS._t[ind_t + 1])
                             <= self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                             for ind_t in range(len(TS._t[:-1]))],
                            names=NamesFormatter.fmt("Min flow", [hub], [hub_2], [self],
                                                     TS._t[:-1]))
                    if max_flow is not None:
                        model_data.cts[self] += model_data.mdl.add_indicators(
                            [model_data.vars["X_N"][(hub, hub_2, self)] for t in TS._t[:-1]],
                            [self._get_F_N(model_data, hub, hub_2, TS._t[ind_t])
                             <= DataAccessors.get2(max_flow, TS._t[ind_t], TS._t[ind_t + 1])
                             for ind_t in range(len(TS._t[:-1]))],
                            names=NamesFormatter.fmt("Max flow", [hub], [hub_2], [self],
                                                     TS._t[:-1]))

                    model_data.cts[self] += model_data.mdl.add_indicators(
                        [model_data.vars["X_N"][(hub, hub_2, self)] for t in TS._t[:-1]],
                        [self._get_F_N(model_data, hub, hub_2, t) <= 0 for t in TS._t[:-1]],
                        true_values=0,
                        names=NamesFormatter.fmt("Status OFF", [hub], [hub_2], [self],
                                                 TS._t[:-1]))


                    if self.get_connection_status(hub, hub_2) == self.no_connection:
                        model_data.cts[self] += [
                            model_data.mdl.add_constraint((model_data.vars["X_N"][(hub, hub_2, self)] == 0),
                                                          ctname=NamesFormatter.fmt_light("No connection", hub, hub_2,
                                                                                          self))]

                    elif self.get_connection_status(hub, hub_2) == self.connection:
                        model_data.cts[self] += [
                            model_data.mdl.add_constraint((model_data.vars["X_N"][(hub, hub_2, self)] == 1),
                                                          ctname=NamesFormatter.fmt_light("Connection", hub, hub_2,
                                                                                          self))]


                    elif self.get_connection_status(hub, hub_2) == self.optim_one_way_max:
                        model_data.cts[self] += [model_data.mdl.add_constraint(
                            (model_data.vars["X_N"][(hub, hub_2, self)] + model_data.vars["X_N"][(hub_2, hub, self)] <= 1),
                            ctname=NamesFormatter.fmt_light("One way connection max", hub, hub_2,
                                                            self))]

                    elif self.get_connection_status(hub, hub_2) == self.optim_one_way_min:
                        model_data.cts[self] += [model_data.mdl.add_constraint(
                            (model_data.vars["X_N"][(hub, hub_2, self)] + model_data.vars["X_N"][(hub_2, hub, self)] >= 1),
                            ctname=
                            NamesFormatter.fmt_light("One way connection min", hub, hub_2,
                                                     self))]

                    if id(hub_2) < id(hub):  # constraints that can be applied only once

                        model_data.cts[self] += [model_data.mdl.add_constraint(
                            (model_data.vars["Y_N"][(hub, hub_2, self)] >= 0.5 * (
                                    model_data.vars["X_N"][(hub_2, hub, self)]
                                  + model_data.vars["X_N"][(hub, hub_2, self)])),
                            ctname=NamesFormatter.fmt_light("One way exists LB", hub, hub_2, self))]

                        model_data.cts[self] += [model_data.mdl.add_constraint(
                            (model_data.vars["Y_N"][(hub, hub_2, self)] <= model_data.vars["X_N"][(hub_2, hub, self)] +
                             model_data.vars["X_N"][(hub, hub_2, self)]),
                            ctname=NamesFormatter.fmt_light("One way exists UP", hub, hub_2, self))]

                        connection_vars += [model_data.vars["Y_N"][(hub, hub_2, self)]]
                    else:
                        connection_vars += [model_data.vars["Y_N"][(hub_2, hub, self)]]

            model_data.cts[self] += [model_data.mdl.add_constraint(
                               len(connection_vars) * model_data.vars["X_SYS"][(hub, self)]
                                >= model_data.mdl.sum_vars(connection_vars),
                            ctname=NamesFormatter.fmt_light("Status hub LB", hub, self)
            )]

            model_data.cts[self] += [model_data.mdl.add_constraint(
                model_data.vars["X_SYS"][(hub, self)]                   # note: the fact that len(connection_vars) is missing is OK
                <= model_data.mdl.sum_vars(connection_vars),
                ctname=NamesFormatter.fmt_light("Status hub UB", hub, self)
            )]


    def plot(self):
        """
        Plots a representation of the network hubs and connections in the (x, y) space.

        A call to this method gives a visual insight of how the network is parametrized, BEFORE optimization.
        The optimization implicitely tranforms every edge status to either 'No connection' or 'Connection'.

        Notes
        -----
        The line linking two hubs is straight for commodity and does not represent
        the real distance function used in the MILP model.


        """
        fig, ax = subplots()
        pos = get_node_attributes(self._network_graph, "coord")
        edges = {edge: self.get_connection_status(edge[0], edge[1])[1]
                 for edge in self._network_graph.edges if self.get_connection_status(edge[0], edge[1]) != self.no_connection}
        edgelist = edges.keys()
        edge_color = edges.values()
        draw_networkx(self._network_graph, pos=pos, edgelist=edgelist, edge_color=edge_color, arrowsize=20, node_color="white", ax=ax)
        [ax.plot([0, 0.0001], [0, 0.0001], linewidth=1, c=color, label=status)  # fake plot to get a legend
         for status, color in [self.connection, self.no_connection, self.optim_one_way_min, self.optim_one_way_max, self.optim_two_ways]
         if (color in edge_color)]
        xmin, xmax = min(v[0] for v in pos.values()), max(v[0] for v in pos.values())
        ymin, ymax = min(v[1] for v in pos.values()), max(v[1] for v in pos.values())
        ax.set_xlim((xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1))       # axis limits are needed to get rid of fake plot
        ax.set_ylim((ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1))
        ax.legend()
        ax.set_title(self)
        return ax