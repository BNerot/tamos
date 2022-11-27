# -** coding: utf-8 -*-
from collections import namedtuple
from datetime import datetime
from docplex.mp.model import Model
from functools import wraps
from itertools import chain
from matplotlib.pyplot import plot, legend, title, tight_layout
from numpy import diff, interp, linspace, mean, argsort, arange, array as array_
from pandas import Series, DataFrame
from scipy.optimize import curve_fit
from tqdm import tqdm
from time import perf_counter

from .component import MetaComponent
from tamos.data_IO.data_IO import DataAccessors, NamesFormatter
from tamos.elementIO import _ElementIO
from .hub import Hub
from tamos.network import _Network
from tamos.production import _Production
from tamos.storage import _Storage

class TimeSettings:
    def __init__(self, n, step_value=1, system_lifetime=40):
        """
        Defines temporal parameters used for the optimization of energy systems.
        Provide helper function to implement a reduced-complexity temporal approach regarding system operation.
        See examples section for in-depth understanding.

        Parameters
        ----------
        n :  int
            The length of the operation period.
        step_value : int, optional, default 1
            Length of each time step of the operation period, in hours.
            Required to be consistent with some component physical properties.
        system_lifetime : int, optional, default 40
            Number of periods considered for economic amortization of components.
            Is related to the property "Discount rate (%)" of components. See method `component.compute_actualized_cost`.
            The default value `40` is a typical value relevant for an annual operation, i.e. such that `n` * `step_value` = 8760

        Examples
        --------
        >>> time_settings = TimeSettings(n=8760, step_value=1, system_lifetime=30)

        The operation period has a length of `n` = 8760.
        Each time serie parameter of components must be of length `n`. Their values last all 1 hour (`step_value` = 1).
        8760 values of 1 hour is one year.
        Economic amortization is calculated for 30 periods, i.e. 30 years.
        Up to now, `time_settings` does not define any relevant time_step for operation. They must be added using the methods:

            * add_regular
            * add_extreme_values
            * add_large_diff

        >>> time_settings.add_regular(5)

        Every index multiple of 5 is selected for operation.
        At this point, the three first time steps (for instance) are defined by the following sets of indexes:

            * {0, 1, 2, 3, 4}
            * {5, 6, 7, 8, 9}
            * {10, 11, 12, 13, 14}
            * ...

        This leads to consider approximately 8760/5 = 1752 time steps for operation.
        Decision variables are indexed on the first element of these index sets.
        Each parameter of the model given as a numpy.ndarray instance of length 8760 is averaged according to these time steps,
        defining a new array `new_array` of length 1752. For instance:

            * new_array[0] = array[0:5].mean()
            * new_array[1] = array[5:10].mean()
            * new_array[2] = array[10:15].mean()
            * ...

        >>> time_settings.add_extreme_values(array=load_values, number_max=3)

        The indexes of the 3 largest values of `load_values` are added to the operation temporal vector, splitting existing time steps.
        For instance, if 2 is one of these 3 indexes, the sets of indexes defining time steps become:

            * {0, 1}
            * {2}
            * {3, 4}
            * {5, 6, 7, 8, 9}
            * {10, 11, 12, 13, 14}
            * ...

        Now, any array will be average according to:

            * new_array[0] = array[0:2].mean()
            * new_array[1] = array[2]
            * new_array[2] = array[3:5].mean()
            * new_array[3] = array[5:10].mean()
            * ...

        >>> time_settings.add_extreme_values(array=temperature_values, number_min=2, number_max=3)

        The indexes of the two smallest and three largest values of `temperature_values` are added to the operation temporal vector.

        """
        self._t0 = 0      # first index of time series being considered
        self._tn = n - 1  # last index of time series being considered
        self._n = n
        self._time_steps = set()
        self._merge_with_existing({0, n-1})      # old: self._merge_with_existing({0})

        self._t_all = range(self._t0, self._tn + 1)
        self._step_value = step_value
        self._system_lifetime = system_lifetime

    @property
    def time_steps(self):
        """
        Presents each time step of the temporal operation vector defined by its set of indexes.

        """
        clean_repr = [f"{self._t[k]}-->{self._t[k+1]-1}" for k in range(len(self._time_steps))]
        return clean_repr

    def prepare_array(self, array, raise_error=False):
        """
        Checks that array length complies with the length of the operation period.
        If array is too long, this method acts depending on `raise_error`.
        If array is too short, an attributeError is raised.

        Parameters
        ----------
        array :  numpy.ndarray
            The array to format.
        raise_error : bool, optional, default False
            Describes how `array` is processed when its length exceeds the length of the operation period:

            * If True, an AttributeError is raised.
            * If False, `array` is sliced so that its first n elements are returned.

        Returns
        -------
        * AttributeError if `raise_error` is True and array is too long, or array is too short.
        * The first n values of `array` otherwise.

        """
        len_ = len(array)
        if len_ > self._n:
            if raise_error:
                raise AttributeError(f"{self}: 'array' is too long. "
                                     f"Expected {self._n} values, got {len_}.")
            else:
                print(f"{self}: 'array' was sliced to comply with n = {self._n}.")
                return array[:self._n]
        elif len_ < self._n:
            raise AttributeError(f"{self}: 'array' is too short. "
                                 f"Expected {self._n} values, got {len_}.")
        else:
            return array_(array)

    def _merge_with_existing(self, indexes, exclusive_integration=False):
        indexes = set(indexes)
        if exclusive_integration:
            indexes |= {e + 1 for e in indexes if e < self._tn}
        self._time_steps |= indexes
        self._t = list(sorted(self._time_steps)) + [self._tn + 1]
        # each dt is closed on the left, the last element behaves as the first, needed for model declaration
        self._dt = [self._t[ind_t + 1] - self._t[ind_t]
                    for ind_t in range(len(self._t[:-1]))]

    def add(self, indexes):
        """
        Adds specific indexes to the operation temporal vector.


        Parameters
        ----------
        indexes : int or list of int
            All values greater than `n` will be ignored.
            Each value of `indexes` defines a new time step,
            i.e. time series parameters of components are not averaged on these indexes.

        """
        try:  # to deal with non iterable 'indexes'
            [_ for _ in indexes]
        except TypeError:
            indexes = [indexes]
        self._merge_with_existing(indexes, exclusive_integration=True)

    def add_extreme_values(self, array, number_min=None, number_max=None):
        """
        Add the time steps corresponding to minimum and maximum values of an array to the operation temporal vector.

        Parameters
        ----------
        array : numpy.ndarray
        number_min, number_max : int, optional
            The indexes of the `number_min` (`numer_max`) smallest (largest) values of `array`
            will be added to the operation temporal vector.

        """
        array = self.prepare_array(array)
        arg_sort_ = argsort(array)
        indexes_min, indexes_max = set(), set()
        if number_min is not None:
            indexes_min = set(arg_sort_[:number_min])
        if number_max is not None:
            indexes_max = set(arg_sort_[-number_max:])
        indexes = indexes_min | indexes_max
        self._merge_with_existing(indexes, exclusive_integration=True)

    def add_large_diff(self, array, number_max):
        """
        Apply 'add_extreme_values' to the absolute value of the temporal derivative of 'array'.

        Parameters
        ----------
        array : numpy.ndarray
        number_max : int
            The indexes of the `number_max` largest values to be added to the operation temporal vector.

        Notes
        -----
        Temporal derivative of `array` is d_array defined as:
            d_array(t) = abs(array(t+1)-array(t))

        """
        array = self.prepare_array(array)
        df = DataFrame(array)
        df["Next step"] = df[0].shift(-1)
        df["Difference"] = abs(df["Next step"] - df[0])
        # last element has no meaning thus it replaced by 0
        # so that the corresponding index is not selected in `add_extreme_values`
        df["Difference"].iloc[-1] = 0
        array_diff = df["Difference"].to_numpy()
        self.add_extreme_values(array_diff, number_min=None, number_max=number_max)

    def add_regular(self, step, offset=0):
        """
        Add every multiple of `step` to the operation temporal vector.

        Parameters
        ----------
        step :  int
        offset : int, >=0, default 0
            Shifts the time steps selection by an amount of `offset` values.

        """
        assert offset >= 0, f"{self}: 'offset' must be positive, got {offset}"
        indexes = arange(self._t0+offset, self._tn, step)
        self._merge_with_existing(indexes, exclusive_integration=False)

    def plot_array_aggregation(self, array):
        """
        Plot the aggregated version of an array, aggregation being performed  according to the `time_steps` attribute.

        This aggregation is how arrays are dealt with in `MILPModel` instances.

        Parameters
        ----------
        array : numpy.ndarray

        """
        array = self.prepare_array(array)
        averaged_array = sum([[DataAccessors.get2(array, self._t[ind_t], self._t[ind_t + 1])]
                           * (self._t[ind_t + 1] - self._t[ind_t])
                          for ind_t in range(len(self._t[:-1]))], [])
        ax = plot(array, label="Original array.")
        ax = plot(averaged_array, label="Array averaged according to the time_steps attribute.")
        # legend()   # fixme uncomment
        tight_layout()
        return ax

    def __repr__(self):
        return f"(n = {self._n}, {len(self._time_steps)} time steps, step value = {self._step_value} hour)"


class ModelData:
    KPI_wrapper = namedtuple("KPI_wrapper", ["DecisionKPI", "kind", "used_in_solving"])

    @staticmethod
    def linear_interpolation(f, a, b, nbr_points=3, nbr_data=1000, plot_=False):
        """
        If problems with np.vectorize (required for some f=lambda x: [...]), try to specify otypes=[float]
        """
        f_a = f(a)
        f_b = f(b)

        def fitting_curve(x_, *args):
            x_break = args[:nbr_points]
            return interp(x_, [a] + list(x_break) + [b], args[nbr_points:])

        x = linspace(a, b, nbr_data)
        y = f(x)
        x_break_0 = linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), nbr_points)
        popt, pcov = curve_fit(fitting_curve, x, y, [*x_break_0, f_a, *f(x_break_0), f_b])
        model_name = "{:d} points".format(nbr_points)
        R_squarred = 1 - sum((fitting_curve(x, *popt) - y) ** 2) / sum((y - mean(y)) ** 2)
        if plot_:
            plot(x, y, label="Function f")
            plot(x, fitting_curve(x, *popt), "-r", label=model_name)
            legend()
            title(r"$R^2=" + "{:.3f}$".format(R_squarred))
        x_break_all = [a] + list(popt)[:nbr_points] + [b]
        y_break_all = popt[nbr_points:]
        slopes = diff(y_break_all) / diff(x_break_all)
        return x_break_all, y_break_all, slopes, R_squarred

    def create_pwlf(self, function, derivative, inf=1e-5, sup=1000, nbr_points=5, nbr_data=1000):
        f = function
        f_der = derivative
        firstslope = f_der(inf)
        lastslope = f_der(sup)
        x_break, y_break, slopes, R_squarred = ModelData.linear_interpolation(f, inf, sup, nbr_points, nbr_data)
        pwlf = self.mdl.piecewise(firstslope, [(x, y) for x, y in zip(x_break, y_break)], lastslope,
                                  name=f"{inf}_{sup}_{nbr_points}")
        return pwlf

    def __init__(self):
        self._reset()
        self.inf = self.mdl.infinity

    def _reset(self):
        self.mdl = Model()
        self.vars = {}
        self.cts = {}
        self.KPIs = {}
        self.var_mapper = {}
        self.KPI_mapper = {}
        self.ct_KPI = {}
        self.total_KPI = {}

    def add_variable_use(self, var_name, var_mapping=None):
        # :param var_mapping:
        #     * None, var is not exported (dataframe, csv, etc)
        #     * tuple of labels, for export
        if var_name in self.vars:  # a var with same name already exists (in keys of mdl.var)
            if var_name in self.var_mapper:
                assert var_mapping == self.var_mapper[
                    var_name]  # if the var is to be exported the labels must comply with existing ones
            else:
                assert var_mapping is None
        else:
            self.vars[var_name] = {}  # new dict attribute named with var_name
            if var_mapping:
                self.var_mapper[var_name] = var_mapping

    def add_KPI_use(self, KPI_name, KPI_mapping):
        if KPI_name in self.KPI_mapper:
            assert KPI_mapping == self.KPI_mapper[KPI_name]
        else:
            self.KPI_mapper[KPI_name] = KPI_mapping


class Description(dict):
    def __setitem__(self, key, value):
        DataAccessors.type_checker(key, "description", "description", str)
        DataAccessors.type_checker(value, "description", "description", (int, float, str))
        super().__setitem__(key, value)


class MILPModel(MetaComponent):
    @staticmethod
    def _sum_nested_iter(dict_):
        """
        :param dict_: dict of nested dicts where base element is an iterable
        :return: all the base elements, with duplicates if exist
        """
        if isinstance(dict_, dict):
            return sum(map(MILPModel._sum_nested_iter, dict_.values()), [])
        return dict_

    def _timer(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            start = perf_counter()
            res = func(self, *args, **kwargs)
            end = perf_counter()
            key = "{:s}.{:s}".format(self.__class__.__name__, func.__name__)
            self._exec_time[key] = end - start
            return res

        return new_func

    def __init__(self, hubs: Hub, time_settings, name=None):
        """
        Manages the declaration and solving processes of the MILP problem associated to the components of hubs `hubs`.


        Parameters
        ----------
        hubs : Hub or list of Hub
            production, storage, element_IO and network components related to `hubs`
            are sized and operated using `time_settings` parameters.
        time_settings : TimeSettings instance
        name : str, optional

        """
        self.name = name
        try:  # to deal with non iterable 'components'
            [_ for _ in hubs]
        except TypeError:
            hubs = [hubs]
        [DataAccessors.type_checker(hub, self, "hubs", Hub) for hub in hubs]
        self._hubs = set(hubs)

        DataAccessors.type_checker(time_settings, self, "time_settings", TimeSettings)
        self._TS = time_settings
        self._model_data = ModelData()
        self._description = Description()
        self._components_assemblies = set()
        self._exec_time = Series({}, dtype=float)
        self._last_solve_ok = False



    @property
    def hubs(self):
        """
        production, storage, element_IO and network components related to `hubs`
        are sized and operated using `time_settings` parameters.

        """
        return self._hubs

    @property
    def description(self):
        """
        Stores a description of the model.
        These properties are used to gather ResultsExport objects written on disk and create ResultsBatch objects.
        They define the `relevant_descriptors` attribute of ResultsBatch objects.

        Some of these properties are automatically declared by tamos. The other are user-defined.
        description: dict

        * keys are str
        * values are int, float or str

        """
        return self._description

    @property
    def name(self):
        """
        Name of this MILPModel instance. If None, name is the current time.

        """
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = datetime.now().strftime("%H_%M_%S")
        self._name = str(name)

    @property
    def components_assemblies(self):
        """
        Components assemblies of the model.

        Must be provided as a list of 3-tuple objects (n_min, n_max, components) where:

            * `n_min` (`n_max`) is the minimum (maximum) number of components from `components` that must be installed, all hubs of `hubs` included.
            * `components` is a component or list of production, storage or element_IO components.
              For any component of `components`, if this component is not in at least one hub of this MILPModel instance,
              the 3-tuple (n_min, n_max, components) is ignored during constraints declaration.

        Examples
        --------
        >>> hub_1 = Hub(components=[heat_load_1, heat_load_2])
        >>> hub_2 = Hub(components=[heat_load_1])
        >>> hub_3 = Hub(components=[heat_load_2])
        >>> MILPModel = MILPModel(hubs=[hub_1, hub_2, hub_3])
        >>> MILPModel.components_assemblies = [(0, 1, heat_load_1), (2, 3, [heat_load_1, heat_load_2])]

        heat_load_1 might be used at most one time, all hubs [hub_1, hub_2, hub_3] included.
        the number of times heat_load_1 or heat_load_2 are used in all hubs [hub_1, hub_2, hub_3]
        is greater than 2 but smaller than 3.

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
                        and 0 <= n_min <= n_max <= len(components) * len(self.hubs)):
                    raise ValueError(f"{self}: Invalid values for 'n_min' and 'n_max'.")

                components = tuple(components)
                objects.append((n_min, n_max, components))
        else:
            objects = components_assemblies
        self._change_set_attribute("_components_assemblies", objects, add=False, remove=False)


    def _declare_components_assemblies_constraints(self):
        pbar = tqdm(self._components_assemblies, desc=f"CTS - [{self}] - assemblies", ncols=120)
        all_types = (_Storage, _Production, _ElementIO, _Network)
        all_attr = ("X_S", "X_P", "X_EXT", "X_SYS")
        model_data = self._model_data
        for idx, (n_min, n_max, components) in enumerate(pbar):
            vars = []
            is_used = False
            for component in components:
                for hub in self.hubs:
                    for component_type, attr in zip(all_types, all_attr):
                        if isinstance(component, component_type):
                            try:
                                vars.append(model_data.vars[attr][(hub, component)])
                                is_used = True
                            except KeyError as e:
                                pass
                if not is_used:
                    print(f"Warning: {self}"
                          f"components_assemblies {(n_min, n_max, components)} "
                          f"involve components that do not apply to any of the hub of {self}."
                          f"Hence they were not declared as constraints.")
                    break
            if is_used:
                if len(vars) >= n_min:
                    model_data.cts[self] += [model_data.mdl.add_constraint(model_data.mdl.sum_vars(vars) <= n_max,
                                                                                 ctname=NamesFormatter.fmt_light(
                                                                                     f"MILPModel assembly n_max", idx))]
                    model_data.cts[self] += [model_data.mdl.add_constraint(model_data.mdl.sum_vars(vars) >= n_min,
                                                                                 ctname=NamesFormatter.fmt_light(
                                                                                     f"MILPModel assembly n_min", idx))]
                else:
                    print(f"Warning: {self}"
                          f"components_assemblies {(n_min, n_max, components)} "
                          f"cannot be satisfied: {len(vars)} components were found in hubs"
                          f"but n_min={n_min}."
                          f"Hence they were not declared as constraints.")

    def _cts_remover(self, cts):
        _ = self._model_data.mdl.remove_constraints(cts)

    def _KPIs_remover(self, KPI_wrappers):
        KPIs = [KPI_wrapper.DecisionKPI for KPI_wrapper in KPI_wrappers]
        for KPI in KPIs:
            _ = self._model_data.mdl.remove_kpi(KPI)

    def remove_constraints_and_KPIs(self, component="all"):
        """
        [experimental, unstable]
        Removes from the MILP model some constraints and KPIs. The next call to `declare_constraints_and_KPIs`
        will only redefine the missing constraints and KPIs, saving model declaration time.
        Relevant after a component modification.

        Parameters
        ----------
        component : Four possible kinds of arg are accepted:

            * 'all': all constraints and KPIs of the MILP model are removed (default)
            * None: assembly_constraints of this MILPModel instance are removed.
              To remove and define constraints on Eco, Exergy and CO2 KPIs, please use the dedicated methods.
            * production, storage, element_IO, or network component: constraints and KPIs of the corresponding component
              are removed, no matter the hub(s) they belong or are related to.
            * hub: constraints and KPIs related to the hub are removed. This includes:

              * hub assemblies constraints
              * constraints and KPIs of all production, storage and element_IO components of hub
              * hub interface constraints (the ones that bind every element exchanges between components)

        Notes
        -----
        1. In case component is a production, storage, element_IO, or network component.
           Constraints and KPIs removal of component does not affect the contribution of the component to the hub interface.
           Unconstrained energy flows will occur if constraints and KPIs are not declared again after call to this method.

        2. Some model modifications require a complete model redeclaration (variables, constraints, KPIs). These cases are:

           * Some components were added or removed from hubs (production, storage, element_IO)
           * New networks are used
           * `time_steps` attribute of `time_settings` was modified

           In these cases, variable redeclaration is automatically performed.

        3. Regarding constraints and KPIs, `Cost` instances are bound to the element_IO component they describe.

        4. Regarding constraints and KPIs, `InterfaceMask` instances are bound to the component they describe.

        """
        if component == "all":
            [self.remove_constraints_and_KPIs(k) for k in self._model_data.cts]
            [self.remove_constraints_and_KPIs(k) for k in self._model_data.KPIs]
        else:
            if component is None:
                component = self
            else:
                DataAccessors.type_checker(component, self, "component", (Hub, _Production, _Storage, _ElementIO, _Network))
            self._remove_constraints_or_KPIs(self._model_data.cts, component, MILPModel._cts_remover)
            self._remove_constraints_or_KPIs(self._model_data.KPIs, component, MILPModel._KPIs_remover)

    def _remove_constraints_or_KPIs(self, dict_, component, remover):
        if component in dict_:
            to_remove = self._sum_nested_iter(dict_[component])
            remover(self, to_remove)
            _ = dict_.pop(component)
        else:
            for k, v in dict_.items():
                if isinstance(v, dict):
                    self._remove_constraints_or_KPIs(v, component, remover)

    def _declare_variables_hubs(self):
        for hub in self._hubs:
            hub._declare_variables(self._model_data, self._TS)

    def _declare_variables_networks(self):
        pbar = tqdm(self._networks, desc="VAR - networks", ncols=120)
        for network in pbar:
            pbar.set_postfix_str(f"{network}")
            network._declare_variables(self._model_data, self._TS)

    def _declare_constraints_self(self):
        if self not in self._model_data.cts:
            self._model_data.cts[self] = []
            self._declare_components_assemblies_constraints()


    def _declare_constraints_hubs(self):
        for hub in self._hubs:
            hub._declare_constraints(self._model_data, self._TS)

    def _declare_constraints_networks(self):
        pbar = tqdm(self._networks, desc="CTS - networks", delay=1e-6, ncols=120)
        for network in pbar:
            if network not in self._model_data.cts:  # empty list of constraints
                pbar.set_postfix_str(f"{network}")
                network._declare_constraints(self._model_data, self._TS)

    def _declare_KPIs(self):
        for hub in self._hubs:
            if hub not in self._model_data.KPIs:
                self._model_data.KPIs[hub] = {}
            hub._declare_KPIs(self._model_data, self._TS)
            # note: hub may exist in model_data.KPIs and have components for which KPIs must be declared again

        pbar = tqdm(self._networks, desc="KPIs - networks", delay=1e-6, ncols=120)
        for network in pbar:
            if network not in self._model_data.KPIs:  # empty list of KPIs
                self._model_data.KPIs[network] = []
                pbar.set_postfix_str(f"{network}")
                network._declare_KPIs(self._model_data, self._TS)
        [self._define_total_KPI(kind) for kind in ["Eco", "CO2", "Exergy"]]

    def _define_total_KPI(self, kind):
        sign = -1 if kind == "Exergy" else 1
        KPI_wrappers = self._sum_nested_iter(self._model_data.KPIs)
        KPIs = [KPI_wrapper.DecisionKPI for KPI_wrapper in KPI_wrappers
                if ((KPI_wrapper.kind == kind) and KPI_wrapper.used_in_solving)]
        self._model_data.total_KPI[kind] = sign * self._model_data.mdl.sum(KPIs)

    def remove_max_KPI_constraint(self, kind):
        """
        Removes the limit on the objective KPI of kind `kind`, if it exists.

        Parameters
        ----------
        kind : {'Eco', 'CO2', 'Exergy'}

        """
        if kind in self._model_data.ct_KPI:
            self._model_data.mdl.remove_constraint(self._model_data.ct_KPI[kind])
            self._model_data.ct_KPI.pop(kind)

    def declare_max_KPI_constraint(self, kind, max_value):
        """
        Constrains one of the objective KPI.
        The values of these objective KPI are described in the `solution_summary` attribute of ResultsExport instances.
        This constraint can be removed by a call to `remove_max_KPI_constraint(kind)`.

        Parameters
        ----------
        kind : {'Eco', 'CO2', 'Exergy'}

        max_value : float

        Notes
        -----
        1. This method calls `remove_max_KPI_constraint(kind)` if necessary to remove any already defined constraint
           regarding kind `kind`.
        2. A constraint on kind `kind` can be defined no matter the kind of objective function.
        3. More than one kind can be constrained at a time.

        Examples
        --------
        >>> MILPModel.declare_max_KPI_constraint("Eco", 1e6)

        The total cost of the system over its lifetime cannot exceed 1 million euros.
        >>> MILPModel.declare_max_KPI_constraint("CO2", 2e5)

        The net CO2 emissions of the system over the operation period (net = entering - exiting) cannot exceed 200 tEqCO2
        >>> MILPModel.declare_max_KPI_constraint("Exergy", 1e5)

        The net exergetical potential consumed by the system over the operation period (net = entering - exiting) cannot exceed 1e5 kWh.
        >>> MILPModel.declare_max_KPI_constraint("Eco", -1e5)

        The system must be economically profitable and generate at least 100 000 euros over its lifetime.

        """
        if not kind in ['Eco', 'CO2', 'Exergy']:
            raise ValueError(f"{self}: 'kind' must be one 'Eco', 'CO2' or 'Exergy'.")
        DataAccessors.type_checker(max_value, self, "max_value", float)
        if kind not in self._model_data.total_KPI:
            raise AssertionError(f"{self}: 'declare_max_KPI_constraint' must be called after the declaration of constraints and KPIs.")
        self.remove_max_KPI_constraint(kind)
        self._model_data.ct_KPI[kind] = self._model_data.mdl.add_constraint(
            self._model_data.total_KPI[kind] <= max_value,
            ctname=f"Max {kind}")

    def _set_objective(self, kind):
        if not kind in ['Eco', 'CO2', 'Exergy']:
            raise ValueError(f"{self}: 'kind' must be one 'Eco', 'CO2' or 'Exergy'.")
        if kind not in self._model_data.total_KPI:
            raise AssertionError(
                f"{self}: 'solve' must be called after the declaration of constraints and KPIs.")
        self.description["Kind"] = kind
        self._model_data.mdl.minimize(self._model_data.total_KPI[kind])

    @_timer
    def solve(self, kind, MIP_gap=1e-4, threads=0, timelimit=3600 * 12):
        """
        Solves the MILP model minimizing the objective function of kind `kind`.
        Model is solved using the Cplex solver. To use another MILP solver, please export the model in LP or MPS format
        using a ResultsExport instance.

        Parameters
        ----------
        kind : {'Eco', 'CO2', 'Exergy'}
        MIP_gap: float, optional, default 1e-4
            0 <= MIG_gap <= 1
            According to Cplex documentation, gap between the best integer objective and the objective of the best node remaining.
        threads: int, optional, default 0
            0 <= threads
            According to Cplex documentation, maximal number of parallel threads that will be invoked by any CPLEX parallel optimizer.

            * If 0, let Cplex decide
            * Else, uses up to `threads` threads

        timelimit: int, optional, default 43200
            0 < timelimit
            According to Cplex documentation, maximum time, in seconds, for a call to an optimizer.

        Notes
        -----
        Other Cplex parameters can be set by accessing the attribute `_model_data.mdl.parameters` of this `MILPModel` instance.
        For instance:
        >>> MILPModel._model_data.mdl.parameters.mip.cuts.flowcovers.set(2)

        """
        self._set_objective(kind)
        self._solve(MIP_gap, threads, timelimit)
        try:
            self._exec_time["Solving deterministic time"] = self._model_data.mdl.solution.solve_details.deterministic_time
        except AttributeError:
            pass

    def _solve(self, MIP_gap, threads, timelimit):
        self._model_data.mdl.parameters.mip.tolerances.mipgap.set(MIP_gap)
        self._model_data.mdl.parameters.threads.set(threads)
        self._model_data.mdl.parameters.timelimit.set(timelimit)
        res = self._model_data.mdl.solve(log_output=True)
        if res is None:
            self._last_solve_ok = False
            print(f"{self}: No solution was found.\n"
                  f"Some details regarding the problems encountered during the solving process\n"
                  f"can be found in the `_model_data.mdl.solve_details` attribute of '{self}'.")
        else:
            self._last_solve_ok = True
            print(f"{self}: Successful solve.")

    def _save_model_state(self):
        self._model_state = {"Components": {hub: hub.components for hub in self._hubs},
                             "networks": self._networks,
                             "_TS._t": self._TS._t}

    def _model_state_checker(self):
        current_components = {hub: hub.components for hub in self._hubs}
        if current_components != self._model_state["Components"]:
            print(
            f"Warning: {self}: some components in hubs were added or removed hence model was reset "
            f"and variables declaration is performed again.")
            return False
        if self._networks != self._model_state["networks"]:
            print(
            f"Warning: {self}: new networks are considered in the system hence model was reset "
            f"and variables declaration is performed again.")
            return False
        if self._TS._t != self._model_state["_TS._t"]:
            print(
            f"Warning: {self}: time_settings was changed hence model was reset "
            f"and variables declaration is performed again.")
            return False
        return True

    def _update_networks(self):
        # all networks related to self._hubs must be included in the MILP problem
        self._networks = set(chain(*[hub.possibly_connected_networks for hub in self._hubs]))


    def _search_networks_orphan_hubs(self):
        # networks whose hubs are not subset of self._hubs must be excluded since network decision variables would be free
        # (due to declaration of network MILP but not hub interface MILP)
        for network in self._networks:
            for hub in network._hubs:
                if network.might_connect(hub) and (hub not in self._hubs):
                    raise AttributeError(f"{self}: hub '{hub}' is able to connect to network '{network}' "
                                         f"yet it is not part of '{self}' hubs. "
                                         f"\nPlease redefine '{self}' "
                                         f"with all hubs used by network '{network}' "
                                         f"or define a 'no_connection' status for hub '{hub}' in network '{network}'.")

    @_timer
    def declare_variables(self):
        """
        Declare all the decision variables associated with components related to `hubs`.

        """
        self._model_data._reset()
        self._update_networks()
        self._save_model_state()
        self._declare_variables_hubs()
        self._declare_variables_networks()

    def _declare_constraints(self):
        self._update_networks()
        self._search_networks_orphan_hubs()
        if not self._model_state_checker():
            self.declare_variables()
        self._declare_constraints_self()
        self._declare_constraints_hubs()
        self._declare_constraints_networks()

    @_timer
    def declare_constraints_and_KPIs(self):
        """
        Declare constraints and KPIs associated with components related to `hubs`.
        All KPIs are defined, i.e. covering each kind of objective function (Eco, Exergy, CO2).

        """
        if not hasattr(self, "_model_state"):
            raise AssertionError(f"{self}: variables must be declared first.")
        else:
            self._declare_constraints()
            self._declare_KPIs()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_model_data" in state:
            del state["_model_data"]
        return state

    def __repr__(self):
        return self.name


