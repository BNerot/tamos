# -*- coding: utf-8 -*-
from tamos.component import Component
from ..data_IO.data_IO import NamesFormatter, DataAccessors


class Production(Component):
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
        The maximum capacity of the component, in kW.
        Relates to decision variable 'SP_P'.
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
        model_data.add_variable_use("F_P", ("Hub", "Element", "Production", "Date"))
        model_data.add_variable_use("X_P", ("Hub", "Production"))
        model_data.add_variable_use("Q_P", ("Hub", "Production", "Date"))
        model_data.add_variable_use("SP_P", ("Hub", "Production"))
        model_data.add_variable_use("N_units")
        used_elements = list(self.used_elements)        # set objects do not preserve order (required for var name consistency)
        model_data.vars["F_P"].update(model_data.mdl.continuous_var_dict(((hub, element, self, t)
                                                                          for element in used_elements
                                                                          for t in TS._t[:-1]),
                                                                         lb=- model_data.inf,
                                                                         name=NamesFormatter.fmt("F_P", [hub],
                                                                                                 used_elements,
                                                                                                 [self], TS._t[:-1])))

        model_data.vars["Q_P"].update(model_data.mdl.continuous_var_dict(((hub, self, t)
                                                                          for t in TS._t[:-1]),
                                                                         name=NamesFormatter.fmt("Q_P", [hub],
                                                                                                 [self], TS._t[:-1])))
        model_data.vars["SP_P"][(hub, self)] = \
            model_data.mdl.continuous_var(name=NamesFormatter.fmt_light("SP_P", hub, self))

        model_data.vars["X_P"][(hub, self)] = \
            model_data.mdl.binary_var(name=NamesFormatter.fmt_light("X_P", hub, self))

        model_data.vars["N_units"][(hub, self)] = model_data.mdl.integer_var(
            name=NamesFormatter.fmt_light("N_units", hub, self))

    def _declare_constraints(self, model_data, hub, TS):
        model_data.cts[hub][self] = []
        model_data.vars["N_units"][(hub, self)].set_ub(self.units_number_ub)
        if self.given_sizing is not None:  # sizing of the component is user-defined
            model_data.cts[hub][self] += [model_data.mdl.add_constraint((model_data.vars["SP_P"][(hub, self)]
                                                                         == model_data.vars["N_units"][
                                                                             (hub, self)] * self.given_sizing),
                                                                        ctname=NamesFormatter.fmt_light(
                                                                            "Capacity definition", hub, self))]
            model_data.cts[hub][self] += [model_data.mdl.add_constraint((model_data.vars["X_P"][(hub, self)] == 1),
                                                                        ctname=NamesFormatter.fmt_light(
                                                                            "Status definition", hub, self))]
        else:
            maxOutputPowerLowerBound = self._get_property("LB max output power (kW)")
            maxOutputPowerUpperBound = self._get_property("UB max output power (kW)")

            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                maxOutputPowerLowerBound * model_data.vars["N_units"][(hub, self)] <=
                model_data.vars["SP_P"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity LB", hub, self))]

            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                model_data.vars["SP_P"][(hub, self)] <=
                maxOutputPowerUpperBound * model_data.vars["N_units"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity UB", hub, self))]


        model_data.cts[hub][self] += [model_data.mdl.add_indicator(model_data.vars["X_P"][(hub, self)],
                                                                   self.units_number_lb <= model_data.vars["N_units"][
                                                                       (hub, self)],
                                                                   name=NamesFormatter.fmt_light("Number LB", hub,
                                                                                                 self))]
        model_data.cts[hub][self] += [model_data.mdl.add_indicator(model_data.vars["X_P"][(hub, self)],
                                                                   model_data.vars["N_units"][(hub, self)] == 0,
                                                                   active_value=0,
                                                                   name=NamesFormatter.fmt_light("Status OFF", hub,
                                                                                                 self))]

        model_data.cts[hub][self] += model_data.mdl.add_constraints((model_data.vars["Q_P"][(hub, self, t)]
                                                                     <= model_data.vars["SP_P"][(hub, self)]
                                                                     for t in TS._t[:-1]),
                                                                    names=NamesFormatter.fmt("Reference flow", [hub],
                                                                                             [self],
                                                                                             TS._t[:-1]))

    def _declare_KPIs(self, model_data, hub, TS):
        if self.eco_count:
            CAPEX = model_data.vars["SP_P"][(hub, self)] * self._get_property("CAPEX (EUR/kW)")

            fixed_OPEX = (self._get_property("OPEX (%CAPEX)") / 100) * CAPEX

            annual_production = model_data.mdl.sum(model_data.vars["Q_P"][(hub, self, TS._t[ind_t])]
                                                   * TS._dt[ind_t]
                                                   for ind_t in range(len(TS._t[:-1]))) * TS._step_value
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
