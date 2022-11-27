# -*- coding: utf-8 -*-
from itertools import chain
from tqdm import tqdm

from .component import MetaComponent
from tamos.data_IO.data_IO import DataAccessors, NamesFormatter
from tamos.element import _ThermalVectorPair, _Element
from tamos.elementIO import _ElementIO
from .mask import InterfaceMask
from tamos.network import _Network
from tamos.production import _Production
from tamos.storage import _Storage


class Hub(MetaComponent):
    """
    Gather components and allow power exchanges through a dedicated interface.

    Enable simple definition of MILP constraints using the `components_assemblies` and `interface_masks` attributes.

    """
    def __init__(self,
                 name=None,
                 components=None,
                 components_assemblies=None,
                 interface_masks=None,
                 ):
        """
        Parameters
        ----------
        name : str, optional
        components: list of storage, production or element_IO instances, optional
        components_assemblies: list of 3-tuple objects (n_min, n_max, components), optional
            Each 3-tuple is such that:

            * `n_min` (`n_max`) is the minimum (maximum) number of components from `components` that must be installed in the hub.
            * `components` is a component or list of production, storage or element_IO components.
               If one component of `components` is not one of hub components,
               the 3-tuple (n_min, n_max, components) is ignored during constraints declaration.

        interface_masks: list of InterfaceMask instances, optional
            Instances in `interface_masks` that do not refer to a component of the hub will be ignored during constraints declaration.

        Examples
        --------
        >>> tms.Hub(components=[heat_load_1, heat_load_2, heat_load_3, electric_heater_1, electric_heater_2, electricity_grid],
        ...         components_assemblies=[(1, 2, [heat_load_1, heat_load_2]), (1, 1, heat_load_3), (0, 1, [electric_heater_1, electric_heater_2])])

        At least one of the first and second loads must be used.
        The third load must be used. At most one of the two heaters must be used.

        >>> tms.Hub(components=[heat_load, electric_heater, electricity_grid, heat_grid],
        ...         components_assemblies=[(0, 1, [electric_heater, heat_grid]), (1, 1, heat_load)])

        The load must be used. Either electric_heater or heat_grid can be used.

        """
        if name is None:
            name = "Hub"
        super().__init__(name)
        self._storages = set()
        self._productions = set()
        self._element_IOs = set()
        self._interface_masks = set()
        self._possibly_connected_networks = set()
        self._components_assemblies = set()

        self.change_components(components)
        self.change_interface_masks(interface_masks)
        self.components_assemblies = components_assemblies

    @property
    def possibly_connected_networks(self):
        """
        network components that include the hub with at least one edge with a connection status is different from `network.no_connection`.

        """
        return self._possibly_connected_networks

    @property
    def components(self):
        """
        production, storage and element_IO components in the hub.

        These can be modified using the `hub.change_components` method.

        """
        return self._productions | self._element_IOs | self._storages

    @property
    def productions(self):
        """
        production components in the hub.

        These can be modified using the `hub.change_components` method.

        """
        return self._productions

    @property
    def element_IOs(self):
        """
        element_IO components in the hub.
        These can be modified using the `hub.change_components` method.

        """
        return self._element_IOs

    @property
    def storages(self):
        """
        storage components in the hub.

        These can be modified using the `hub.change_components` method.

        """
        return self._storages

    def change_components(self, components, add=False, remove=False):
        """
        Change the components that can be installed in the hub.

        Parameters
        ----------
        components: instance or list of instances of production, storage and element_IO components.
        add : bool, optional
            If True, `components` are added to already defined components in hub.
        remove : bool, optional
            If True, all instances from `components` that are components of the hub are removed from the hub.

        Notes
        -----
        If `add` and `remove` are False, hub components are replaced by `components`.
        `add` and `remove` cannot be both True.

        """
        if all([add, remove]):
            raise ValueError(f"{self}: Cannot simultaneously add and remove components.")

        if components is not None:
            try:  # to deal with non iterable 'components'
                [_ for _ in components]
            except TypeError:
                components = [components]
            for component in components:
                if not isinstance(component, (_Storage, _Production, _ElementIO)):
                    raise TypeError(
                        f"{self}: Invalid data type for component '{component}' in 'components'.")

            for components_attr_name, type_ in zip(["_storages", "_productions", "_element_IOs"],
                                                  [_Storage, _Production, _ElementIO]):
                components_ = filter(lambda arg: isinstance(arg, type_), components)
                self._change_set_attribute(components_attr_name, components_, add=add, remove=remove)
        else:
            for components_attr_name in ["_storages", "_productions", "_element_IOs"]:
                self._change_set_attribute(components_attr_name, components, add=add, remove=remove)

    @property
    def interface_masks(self):
        """
        InterfaceMask instances of the hub.

        These can be modified using the `hub.change_interface_masks` method.

        """
        return self._interface_masks

    def change_interface_masks(self, interface_masks, add=False, remove=False):
        """
        Change the InterfaceMasks instances related to the hub.

        Parameters
        ----------
        interface_masks: instance or list of instances of InterfaceMask.
            Instances in `interface_masks` that do not refer to a component of the hub will be ignored during constraints declaration.
        add : bool, optional
            If True, `components` are added to already defined components in hub.
        remove : bool, optional
            If True, all instances from `components` that are components of the hub are removed from the hub.

        Notes
        -----
        If `add` and `remove` are False, hub components are replaced by `components`.
        `add` and `remove` cannot be both True.

        """
        if all([add, remove]):
            raise ValueError(f"{self}: Cannot simultaneously add and remove interface masks.")
        if interface_masks is not None:
            try:  # to deal with non iterable 'interface_masks'
                [_ for _ in interface_masks]
            except TypeError:
                interface_masks = [interface_masks]
            for interface_mask in interface_masks:
                if not isinstance(interface_mask, InterfaceMask):
                    raise TypeError(
                        f"{self}: Invalid data type for interface_mask '{interface_mask}' in 'interface_masks'.")
        self._change_set_attribute("_interface_masks", interface_masks, add=add, remove=remove)

    @property
    def components_assemblies(self):
        """
        Components assemblies of the hub.

        Must be provided as a list of 3-tuple objects (n_min, n_max, components) where:

        * `n_min` (`n_max`) is the minimum (maximum) number of components from `components` that must be installed in the hub.
        * `components` is a component or list of production, storage or element_IO components. If one component of `components` is not one of hub components,
          the 3-tuple (n_min, n_max, components) is ignored during constraints declaration.

        Examples
        --------
        >>> hub.components_assemblies = [(1, 2, [heat_load_1, heat_load_2]), (1, 1, heat_load_3), (0, 1, [electric_heater_1, electric_heater_2])])

        At least one of the first and second loads must be used.
        The third load must be used. At most one of the two heaters must be used.

        >>> hub.components_assemblies = [(0, 1, [electric_heater, heat_grid]), (1, 1, heat_load)])

        The load must be used. Either electric_heater or heat_grid can be used.

        """
        return self._components_assemblies


    @components_assemblies.setter
    def components_assemblies(self, components_assemblies):
        if components_assemblies is not None:
            all_types = (_Storage, _Production, _ElementIO, _Network)
            objects = []
            for x in components_assemblies:
                try:
                    *y, components = x
                except Exception:
                    raise TypeError(f"{self}: Invalid content of 'components_assemblies'.")
                try:  # to deal with non iterable 'components'
                    [_ for _ in components]
                except TypeError:
                    components = [components]
                for component in components:
                    if not isinstance(component, all_types):
                        raise TypeError(f"{self}: Invalid data type for components '{components}' "
                                        f"in 'components_assemblies'.")
                try:
                    n_min, n_max = y
                except Exception:
                    raise TypeError(f"{self}: Invalid content of 'components_assemblies'.")
                if not (isinstance(n_min, int) and isinstance(n_max, int)
                        and 0 <= n_min <= n_max <= len(components)):
                    raise ValueError(f"{self}: Invalid values for 'n_min' and 'n_max'.")

                components = tuple(components)
                objects.append((n_min, n_max, components))
        else:
            objects = components_assemblies
        self._change_set_attribute("_components_assemblies", objects, add=False, remove=False)



    def _get_used_elements_storage(self):
        return set(chain(*[storage.used_elements for storage in self._storages]))

    def _get_used_elements_production(self):
        return set(chain(*[production.used_elements for production in self._productions]))

    def _get_used_elements_element_IO(self):
        return set(chain(*[element_IO.used_elements for element_IO in self._element_IOs]))

    def _get_intra_hub_used_elements(self):
        return self._get_used_elements_storage() | \
               self._get_used_elements_production() | \
               self._get_used_elements_element_IO()


    def _declare_variables(self, model_data, TS):
        for components, desc in [(self._storages, "storages"),
                                 (self._productions, "productions"),
                                 (self._element_IOs, "element_IOs"),
                                 ]:
            pbar = tqdm(components, desc=f"VAR - [{self}] - {desc}", ncols=120)
            for component in pbar:
                pbar.set_postfix_str(f"{component}")
                component._declare_variables(model_data, self, TS)

    def _declare_constraints(self, model_data, TS):
        if self not in model_data.cts:
            self._declare_hub_constraints(model_data, TS)

        self._declare_components_constraints(model_data, TS)

        pbar = tqdm(self.interface_masks, desc=f"CTS - [{self}] - interface_masks", ncols=120)
        for interface_mask in pbar:
            if interface_mask.component in self.possibly_connected_networks | self.components:
                interface_mask._declare_constraints(model_data, self, TS)
            else:
                print(f"Warning: {self}: interface_mask '{interface_mask}' involves a component unused by hub, "
                      f"hence it was not applied.")

    def _declare_components_constraints(self, model_data, TS):

        for components, desc in [(self._storages, "storages"),
                                 (self._productions, "productions"),
                                 (self._element_IOs, "element_IOs"),
                                 ]:
            pbar = tqdm(components, desc=f"CTS - [{self}] - {desc}", ncols=120)
            for component in pbar:
                if component not in model_data.cts[self]:  # empty list of constraints
                    pbar.set_postfix_str(f"{component}")
                    component._declare_constraints(model_data, self, TS)

    def _declare_hub_constraints(self, model_data, TS):
        model_data.cts[self] = {self: []}
        self._declare_interface_constraints(model_data, TS)
        self._declare_components_assemblies_constraints(model_data, TS)

    def _get_flow_var(self, flow_var_dict, component, element, t1, multi_elements):
        if multi_elements:
            return flow_var_dict[(self, element, component, t1)]
        else:
            return flow_var_dict[(self, component, t1)]

    def _declare_interface_constraints(self, model_data, TS):
        networks = self.possibly_connected_networks
        used_elements_network = set(chain(*[network.used_elements for network in networks]))
        all_elements = self._get_intra_hub_used_elements() | used_elements_network
        # ordered collection for naming
        vectors = list(_Element._decompose_elements(all_elements))
        pbar = tqdm(vectors, desc=f"CTS - [{self}] - interface", ncols=120)
        for vector in pbar:
            pbar.set_postfix_str(f"{vector}")
            total = {}
            for components, var_name, name, multi_elements in \
                                              zip([self._storages, self._productions, self._element_IOs, networks],
                                                  ["F_S", "F_P", "F_EXT", "F_SYS"],
                                                  ["Storage", "Production", "ElementIO", "Network"],
                                                  [True, True, False, False]):
                var = model_data.vars.get(var_name, None)
                total[name] = [0 for ind_t in range(len(TS._t[:-1]))]
                for component in components:
                    for element in component.used_elements:
                        if isinstance(element, _ThermalVectorPair):
                            if element._out_TV == vector:
                                for ind_t in range(len(TS._t[:-1])):
                                    t1, t2 = TS._t[ind_t], TS._t[ind_t + 1]
                                    CpDT_ = DataAccessors.get2(element._CpDT, t1, t2)
                                    var_ = self._get_flow_var(var, component, element, t1, multi_elements)
                                    if CpDT_ != 0:
                                        total[name][ind_t] += var_ / CpDT_
                                    else:
                                        pass

                            elif element._in_TV == vector:
                                for ind_t in range(len(TS._t[:-1])):
                                    t1, t2 = TS._t[ind_t], TS._t[ind_t + 1]
                                    CpDT_ = DataAccessors.get2(element._CpDT, t1, t2)
                                    var_ = self._get_flow_var(var, component, element, t1, multi_elements)
                                    if CpDT_ != 0:
                                        total[name][ind_t] += -var_ / CpDT_
                                    else:
                                        pass
                            else:
                                pass
                        else:
                            if element == vector:
                                for ind_t in range(len(TS._t[:-1])):
                                    t1, t2 = TS._t[ind_t], TS._t[ind_t + 1]
                                    var_ = self._get_flow_var(var, component, element, t1, multi_elements)
                                    total[name][ind_t] += var_

            cts = [model_data.mdl.eq_constraint(total["Network"][ind_t] + total["Production"][ind_t]
                                                 + total["Storage"][ind_t] + total["ElementIO"][ind_t], 0)
                    for ind_t in range(len(TS._t[:-1]))]
            names = NamesFormatter.fmt("INTERFACE", [self], [vector], TS._t[:-1])
            model_data.cts[self][self] += model_data.mdl.add_constraints(cts, names=names)
            # no use of eq overload "==" to deal with the case all(total.values()) are 0 that yields "True"


    def _declare_components_assemblies_constraints(self, model_data, TS):
        pbar = tqdm(self._components_assemblies, desc=f"CTS - [{self}] - assemblies", ncols=120)
        used_components = self.components
        used_networks = self.possibly_connected_networks
        all_types = (_Storage, _Production, _ElementIO, _Network)
        all_attr = ("X_S", "X_P", "X_EXT", "X_SYS")
        for idx, (n_min, n_max, components) in enumerate(pbar):
            cond = True
            for component in components:
                if isinstance(component, (_Production, _Storage, _ElementIO)):
                    cond_ = component in used_components
                else:
                    assert isinstance(component, _Network)
                    cond_ = component in used_networks
                cond &= cond_
            if cond:
                vars = []
                for component in components:
                    for component_type, attr in zip(all_types, all_attr):
                        if isinstance(component, component_type):
                            vars.append(model_data.vars[attr][(self, component)])
                if n_min == n_max == len(components):
                    model_data.cts[self][self] += model_data.mdl.add_constraints(
                        (var == 1 for var in vars),
                        names=NamesFormatter.fmt(f"Hub assembly", [self], [idx], components)
                    )
                else:
                    model_data.cts[self][self] += [model_data.mdl.add_constraint(model_data.mdl.sum_vars(vars) <= n_max,
                                                                                 ctname=NamesFormatter.fmt_light(
                                                                                     f"Hub assembly n_max", self, idx))]
                    model_data.cts[self][self] += [model_data.mdl.add_constraint(model_data.mdl.sum_vars(vars) >= n_min,
                                                                                 ctname=NamesFormatter.fmt_light(
                                                                                     f"Hub assembly n_min", self, idx))]
            else:
                print(f"Warning: {self}"
                      f"components_assemblies {(n_min, n_max, components)} "
                      f"involve components that do not apply to hub."
                      f"Hence they were not declared as constraints.")



    def _declare_KPIs(self, model_data, TS):
        for components, desc in [(self._storages, "storages"),
                                 (self._productions, "productions"),
                                 (self._element_IOs, "element_IOs")]:
            pbar = tqdm(components, desc=f"KPIs - [{self}] - {desc}", ncols=120, delay=1e-6)
            for component in pbar:
                if component not in model_data.KPIs[self]:
                    if desc == "element_IOs":
                        model_data.KPIs[self][component] = {}
                    else:
                        model_data.KPIs[self][component] = []
                    pbar.set_postfix_str(f"{component}")
                    component._declare_KPIs(model_data, self, TS)

    def describe(self):
        """
        Show an exhaustive description of the hub.

        """
        print(self.name)
        for attr_name, public_attr_name in \
            zip(
                ["productions", "storages", "element_IOs", "possibly_connected_networks",
                 "components_assemblies", "interface_masks"],
                ["Productions", "Storages", "Element_IOs", "Possibly connected networks",
                 "Components assemblies", "Interface masks"]
            ):
            attr = getattr(self, attr_name)
            if attr:
                print(public_attr_name, ":\n", attr, "\n")
