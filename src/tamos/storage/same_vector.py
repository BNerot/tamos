from tamos.data_IO.data_IO import NamesFormatter
from tamos.element import ElectricityVector, FuelVector
from .storage_base import Storage


class OneVector(Storage):
    def __init__(self, vector, properties, given_sizing=None, name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        OneVector components store Electricity or FuelVector elements.

        This component declares the following exported decision variables:

        * X_S, binary.
          Whether the component is used by the hub.
        * SE_S, continuous, in kWh.
          The maximum energetical capacity of the component.
        * For all t, E_S(t), continuous, in kWh.
          State of charge of the storage.
        * For all t, for all element e, F_S(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).

        This component declares the following KPIs:

        * `COST_storage`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        vector : ElectricityVector or FuelVector
            Stored element.
        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max energy (kWh)"
            * "UB max energy (kWh)"
            * "Charge/discharge delay (h)"
            * "Energy conservation (/h)"
            * "CAPEX energy (EUR/kWh)"
            * "OPEX energy (%CAPEX)"

        given_sizing : int or float, optional
            The maximum capacity of the component, in kWh.
            Relates to decision variable 'SE_S'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        name : str, optional
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max energy (kWh)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        """
        if name is None:
            name = f"SameVector..{vector}"
        super().__init__(properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self._add_used_element(vector, "element", (ElectricityVector, FuelVector))

    @property
    def vector(self):
        """
        Stored element

        """
        return self._element

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        if self.given_sizing is None:
            maxEnergyLowerBound = self._get_property("LB max energy (kWh)")
            maxEnergyupperBound = self._get_property("UB max energy (kWh)")


            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                maxEnergyLowerBound * model_data.vars["N_units"][(hub, self)] <=
                model_data.vars["SE_S"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity LB", hub, self))]

            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                model_data.vars["SE_S"][(hub, self)] <=
                maxEnergyupperBound * model_data.vars["N_units"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity UB", hub, self))]

        vector = self.element
        charge_discharge_delay = self._get_property("Charge/discharge delay (h)")
        energy_conservation = self._get_property("Energy conservation (/h)")

        max_charging_discharging = model_data.vars["SE_S"][(hub, self)] / charge_discharge_delay
        model_data.cts[hub][self] += model_data.mdl.add_constraints((- max_charging_discharging
                                                                     <= model_data.vars["F_S"][(hub, vector, self, t)]
                                                                     for t in TS._t[:-1]),
                                                                    names=NamesFormatter.fmt("Discharge UB", [hub],
                                                                                             [self], TS._t[:-1]))

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
                                    (model_data.vars["F_S"][(hub, vector, self, t)]
                                     <= max_charging_discharging
                                     for t in TS._t[:-1]),
                                     names=NamesFormatter.fmt("Charge UB", [hub], [self], TS._t[:-1])
                                                            )

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["E_S"][(hub, self, TS._t[ind_t + 1])]
             == energy_conservation ** (TS._dt[ind_t] * TS._step_value)
             * model_data.vars["E_S"][(hub, self, TS._t[ind_t])]
             + model_data.vars["F_S"][(hub, self._element, self, TS._t[ind_t])]
             * TS._dt[ind_t] * TS._step_value
             for ind_t in range(len(TS._t) - 2)),
            names=NamesFormatter.fmt_light("SOC balance", hub, self, TS._t[:-2]))

        model_data.cts[hub][self] += [model_data.mdl.add_constraint((model_data.vars["E_S"][(hub, self, TS._t[0])]
                                                     == energy_conservation ** ((TS._t[-1] - TS._t[-2]) * TS._step_value)
                                                      * model_data.vars["E_S"][(hub, self, TS._t[-2])]
                                                      + model_data.vars["F_S"][(hub, self._element, self, TS._t[-2])]
                                                      * (TS._t[-1] - TS._t[-2]) * TS._step_value),        # TS._t[-1] - TS._t[-2] is TS._dt[-1]
                                           ctname=NamesFormatter.fmt_light("SOC boundaries", hub, self, TS._t[-2]))]

    def _declare_KPIs(self, model_data, hub, TS):
        if self.eco_count:
            CAPEX = model_data.vars["SE_S"][(hub, self)] * self._get_property("CAPEX energy (EUR/kWh)")
            fixed_OPEX = (self._get_property("OPEX energy (%CAPEX)") / 100) * CAPEX
            cost_storage, CAPEX__, OPEX__ = self.compute_actualized_cost(CAPEX,
                                                                         fixed_OPEX,
                                                                         TS._system_lifetime)
            base_name = "COST_storage"
            for name_, value, used_in_solving in [(f"{base_name} - CAPEX", CAPEX__, False),
                                                  (f"{base_name} - OPEX", OPEX__, False),
                                                  (f"{base_name}", cost_storage, True),
                                                  ]:
                model_data.add_KPI_use(name_, ("Hub", "Storage"))
                name = NamesFormatter.fmt_light(name_, hub, self, KPI=True)
                kpi = model_data.mdl.add_kpi(value, publish_name=name)
                model_data.KPIs[hub][self] += [model_data.KPI_wrapper(kpi, "Eco", used_in_solving)]
