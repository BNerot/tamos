from .component import MetaComponent
from .data_IO.data_IO import DataAccessors, NamesFormatter
from tamos.elementIO import _ElementIO
from tamos.network import _Network
from tamos.production import _Production
from tamos.storage import _Storage


class InterfaceMask(MetaComponent):
    def __init__(self,
                 component,
                 element=None,
                 name=None,
                 power_lb=None,
                 power_ub=None,
                 energy_lb=None,
                 energy_ub=None):
        """
        Defines power and energy constraints regarding exchanges between a component and the hub it is affected to.

        Must be passed to a hub to be effective.
        An InterfaceMask instance can be affected to different hubs. Constraints are applied in an independent manner
        no matter the hubs it is affected to.

        Parameters
        ----------
        component: production, storage, element_IO or network instance
        element: element instance, optional
            Must be provided only if `component` is a production or storage instance.
        name: str, optional
        power_lb, power_ub: numpy.ndarray or float, optional
            The lower bound (upper bound) of the power related to the element `element` (if relevant) of component `component`.
        energy_lb, energy_ub: float, optional
            The lower bound (upper bound) of the energy related to the element `element` (if relevant) of component `component`.
            Energy is calculated as sum(power(t) * dt(t)) on the entire operation period.

        Examples
        --------
        >>> tms.InterfaceMask(component=electric_heater, element=electricity, energy_ub=1000)

        The amount of energy related to element `electricity` during the entire operation period must not exceed 1000 kWh.

        >>> tms.InterfaceMask(component=electric_heater, element=heat, power_lb=5)

        The amount of energy related to element `heat` must be always greater than 5 kW.

        >>> tms.InterfaceMask(component=electric_heater, element=heat, power_lb=numpy.linspace(1, 10, 8760),
        ...                                                            power_ub=numpy.linspace(1, 10, 8760))

        The amount of energy related to element `heat` must increase linearly with time, from 1 to 10 kW.

        """

        if name is None:
            name = f"InterfaceMask..{component}..{element}"
        super().__init__(name)
        self._compatibility_check(component, element)
        self.power_lb = power_lb
        self.power_ub = power_ub
        self.energy_lb = energy_lb
        self.energy_ub = energy_ub
        self._prepare_vars()

    @property
    def component(self):
        """
        The component whose element exchanges with the hub interface are constrained.

        """
        return self._component

    @property
    def element(self):
        """
        If `component` is a storage or production instance, the element whose power flows and energy balances are constrained.

        """
        return self._element

    @property
    def power_lb(self):
        """
        The lower bound of the power related to the element `element` (if relevant) of component `component`.
        None, numpy.ndarray or float.

        """
        return self._power_lb

    @power_lb.setter
    def power_lb(self, power_lb):
        if power_lb is not None:
            DataAccessors.type_checker(power_lb, self, "power_lb", "numeric")
        self._power_lb = power_lb

    @property
    def power_ub(self):
        """
        The upper bound of the power related to the element `element` (if relevant) of component `component`.
        None, numpy.ndarray or float.

        """
        return self._power_ub

    @power_ub.setter
    def power_ub(self, power_ub):
        if power_ub is not None:
            DataAccessors.type_checker(power_ub, self, "power_ub", "numeric")
        self._power_ub = power_ub

    @property
    def energy_lb(self):
        """
        The lower bound of the energy related to the element `element` (if relevant) of component `component`.
        Energy is calculated as sum(power(t) * dt(t)) on the entire optimization period.
        None or float.

        """
        return self._energy_lb

    @energy_lb.setter
    def energy_lb(self, energy_lb):
        if energy_lb is not None:
            DataAccessors.type_checker(energy_lb, self, "energy_lb", "numeric")
        self._energy_lb = energy_lb

    @property
    def energy_ub(self):
        """
        The upper bound of the energy related to the element `element` (if relevant) of component `component`.
        Energy is calculated as sum(power(t) * dt(t)) on the entire optimization period.
        None or float.

        """
        return self._energy_ub

    @energy_ub.setter
    def energy_ub(self, energy_ub):
        if energy_ub is not None:
            DataAccessors.type_checker(energy_ub, self, "energy_ub", "numeric")
        self._energy_ub = energy_ub

    def _compatibility_check(self, component, element):
        if isinstance(component, (_Network, _ElementIO)):
            assert (element is None) or (element is component.element), f"{self}: element does not match component."
            element = component.element
            multi_elements = False

        elif isinstance(component, (_Production, _Storage)):
            if element is None:
                raise AttributeError(f"{self}: an element must be specified with component {component}.")
            assert element in component._used_elements, f"{self}: component {component} does not use element {element}."
            multi_elements = True

        else:
            raise TypeError(f"{self}: Component {component} is not of a valid type. "
                            f"Expected one of: {_Production}, {_Storage}, {_Network}, {_ElementIO}.")
        self._component = component
        self._element = element
        self._multi_elements = multi_elements

    def _prepare_vars(self):
        if self._multi_elements:
            if isinstance(self.component, _Storage):
                flow_var_name = "F_S"
                status_var_name = "X_S"
            else:
                flow_var_name = "F_P"
                status_var_name = "X_P"
        else:
            if isinstance(self.component, _Network):
                flow_var_name = "F_SYS"
                status_var_name = "X_SYS"
            else:
                flow_var_name = "F_EXT"
                status_var_name = "X_EXT"

        self._flow_var_name = flow_var_name
        self._status_var_name = status_var_name

    def _get_flow_var(self, flow_var_dict, hub, t):
        # might be under optimal in term of declaration time
        if self._multi_elements:
            return flow_var_dict[(hub, self._element, self._component, t)]
        else:
            return flow_var_dict[(hub, self._component, t)]

    def _declare_constraints(self, model_data, hub, TS):
        flow_var_dict = model_data.vars[self._flow_var_name]
        status_var = model_data.vars[self._status_var_name][(hub, self.component)]

        cts = []
        # power bounds
        base_name = "Power"
        status_var_ = [status_var for _ in TS._t[:-1]]

        if self.power_lb is not None:

            cts += model_data.mdl.add_indicators(
                status_var_,
                (DataAccessors.get2(self._power_lb, TS._t[ind_t], TS._t[ind_t + 1])
                 <= self._get_flow_var(flow_var_dict, hub, TS._t[ind_t])
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt(f"{base_name} LB", [hub], [self.component], [self], TS._t[:-1]))

        if self.power_ub is not None:
            cts += model_data.mdl.add_indicators(
                status_var_,
                (self._get_flow_var(flow_var_dict, hub, TS._t[ind_t])
                 <= DataAccessors.get2(self._power_ub, TS._t[ind_t], TS._t[ind_t + 1])
                 for ind_t in range(len(TS._t[:-1]))),
                names=NamesFormatter.fmt(f"{base_name} UB", [hub], [self.component], [self], TS._t[:-1])
            )

        # energy bounds
        base_name = "Energy"
        annual_energy = model_data.mdl.sum(self._get_flow_var(flow_var_dict, hub, TS._t[ind_t])
                                           * TS._dt[ind_t]
                                           for ind_t in range(len(TS._t[:-1]))) * TS._step_value

        if self.energy_lb is not None:
            cts += [model_data.mdl.add_indicator(
                status_var,
                self.energy_lb <= annual_energy,
                name=NamesFormatter.fmt_light(f"{base_name} LB", hub, self.component, self)
            )]

        if self.energy_ub is not None:
            cts += [model_data.mdl.add_indicator(
                status_var,
                annual_energy <= self.energy_ub,
                name=NamesFormatter.fmt_light(f"{base_name} UB", hub, self.component, self)
            )]

        if isinstance(self.component, _Network):
            model_data.cts[hub][hub] += cts
        else:
            model_data.cts[hub][self.component] += cts
