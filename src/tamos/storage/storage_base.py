# -*- coding: utf-8 -*-
from tamos.component import Component
from tamos.data_IO.data_IO import DataAccessors, NamesFormatter


class Storage(Component):
    def __init__(self, properties, name, given_sizing, units_number_ub, units_number_lb, eco_count):
        super().__init__(name, properties)
        self.units_number_ub = units_number_ub
        self.units_number_lb = units_number_lb
        self.given_sizing = given_sizing
        self.eco_count = eco_count



    @property
    def units_number_lb(self):
        """
        The lower bound of the number of real components that this instance aims to stand for.
        Setting `units_number_lb` has a meaning if "LB max output power (kW)" property is different from 0.
        int

        """
        return self._units_number_lb

    @units_number_lb.setter
    def units_number_lb(self, units_number_lb):
        DataAccessors.type_checker(units_number_lb, self, "units_number_lb", int)
        if hasattr(self, "_units_number_ub"):  # for __init__
            assert self.units_number_ub >= units_number_lb, \
                f"{self}: 'units_number_lb' must be lower or equal than 'units_number_ub'."
        self._units_number_lb = units_number_lb

    @property
    def units_number_ub(self):
        """
        The upper bound of the number of real components that this instance aims to stand for.
        Setting `units_number_ub` has a meaning if "LB max output power (kW)" property is different from 0.
        int

        """
        return self._units_number_ub

    @units_number_ub.setter
    def units_number_ub(self, units_number_ub):
        DataAccessors.type_checker(units_number_ub, self, "units_number_ub", int)
        if hasattr(self, "_units_number_lb"):  # for __init__
            assert units_number_ub >= self.units_number_lb, \
                f"{self}: 'units_number_ub' must be higher or equal than 'units_number_lb'."
        self._units_number_ub = units_number_ub

    @property
    def given_sizing(self):
        """
        The maximum capacity of the component, in kg (Thermocline) or kWh (OneVector).
        Relates to decision variable 'SE_S'.
        If specified, only the operation of this component is performed by the MILP solver.
        If let unknown, both sizing and operation are performed.
        int or float

        """
        return self._given_sizing

    @given_sizing.setter
    def given_sizing(self, given_sizing):
        if given_sizing is not None:
            DataAccessors.type_checker(given_sizing, self, "given_sizing", (int, float))
        self._given_sizing = given_sizing

    @property
    def eco_count(self):
        """
        Whether this instance contributes to the system "Eco" KPI.
        bool

        """
        return self._eco_count

    @eco_count.setter
    def eco_count(self, eco_count):
        DataAccessors.type_checker(eco_count, self, "eco_count", bool)
        self._eco_count = eco_count

    def _declare_variables(self, model_data, hub, TS):

        model_data.add_variable_use("F_S", ("Hub", "Element", "Storage", "Date"))
        model_data.add_variable_use("X_S", ("Hub", "Storage"))
        model_data.add_variable_use("E_S", ("Hub", "Storage", "Date"))
        model_data.add_variable_use("SE_S", ("Hub", "Storage"))
        model_data.add_variable_use("N_units")

        used_elements = list(
            self.used_elements)  # set objects do not preserve order (required for var name consistency)

        model_data.vars["F_S"].update(model_data.mdl.continuous_var_dict(((hub, element, self, t)
                                                                          for element in used_elements
                                                                          for t in TS._t[:-1]),
                                                                         lb=- model_data.inf,
                                                                         name=NamesFormatter.fmt("F_S", [hub],
                                                                                                 used_elements,
                                                                                                 [self], TS._t[:-1])))
        model_data.vars["X_S"][(hub, self)] = \
            model_data.mdl.binary_var(name=NamesFormatter.fmt_light("X_S", hub, self))

        model_data.vars["E_S"].update(model_data.mdl.continuous_var_dict(((hub, self, t) for t in TS._t[:-1]),
                                                                         name=NamesFormatter.fmt("E_S", [hub], [self],
                                                                                                 TS._t[:-1])))

        model_data.vars["SE_S"][(hub, self)] = \
            model_data.mdl.continuous_var(name=NamesFormatter.fmt_light("SE_S", hub, self))

        model_data.vars["N_units"][(hub, self)] = model_data.mdl.integer_var(
            name=NamesFormatter.fmt_light("N_units", hub, self))

    def _declare_constraints(self, model_data, hub, TS):
        model_data.cts[hub][self] = []
        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["E_S"][(hub, self, t)] <= model_data.vars["SE_S"][(hub, self)] for t in TS._t[:-1]),
            names=NamesFormatter.fmt("SOC UB", [hub], [self], TS._t[:-1]))

        model_data.vars["N_units"][(hub, self)].set_ub(self.units_number_ub)

        if self.given_sizing is not None:
            model_data.cts[hub][self] += [model_data.mdl.add_constraint(model_data.vars["SE_S"][(hub, self)]
                                                                        == model_data.vars["N_units"][
                                                                            (hub, self)] * self.given_sizing,
                                                                        ctname=NamesFormatter.fmt_light(
                                                                            "Capacity definition", hub,
                                                                            self))]
            model_data.cts[hub][self] += [model_data.mdl.add_constraint((model_data.vars["X_S"][(hub, self)] == 1),
                                                                        ctname=NamesFormatter.fmt_light(
                                                                            "Status definition", hub, self))]

        else:
            pass

        model_data.cts[hub][self] += [model_data.mdl.add_indicator(model_data.vars["X_S"][(hub, self)],
                                                                   self.units_number_lb <= model_data.vars["N_units"][
                                                                       (hub, self)],
                                                                   name=NamesFormatter.fmt_light("Number LB", hub,
                                                                                                 self))]
        model_data.cts[hub][self] += [model_data.mdl.add_indicator(model_data.vars["X_S"][(hub, self)],
                                                                   model_data.vars["N_units"][(hub, self)] == 0,
                                                                   active_value=0,
                                                                   name=NamesFormatter.fmt_light("Status OFF", hub,
                                                                                                 self))]
