from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair
from .storage_base import Storage


class Thermocline(Storage):

    def __init__(self, stored_TVP, properties=None,
                 given_sizing=None, units_number_ub=1, units_number_lb=1, name=None, eco_count=True):
        """
        Thermocline components model a perfectly stratified thermal energy storage.

        This component declares the following exported decision variables:

        * X_S, binary.
          Whether the component is used by the hub.
        * SE_S, continuous, in kg.
          The maximum energetical capacity of the component.
        * For all t, E_S(t), continuous, in kg.
          State of charge of the storage.
        * For all t, for all element e, F_S(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).

        This component declares the following KPIs:

        * `COST_storage`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        stored_TVP : ThermalVectorPair
            Stored element.
            Storage is full when its entire content is the incoming vector of `stored_TVP`.
            Please see note 2) for an explanation of the restriction regarding `stored_TVP`.
        properties : dict {str: int | float}
            Techno-economic properties of the component.

            The `properties` attribute must include the following keys:

            * "LB max energy (kg)"
            * "UB max energy (kg)"
            * "Charge/discharge delay (h)"
            * "Energy conservation (/h)"
            * "CAPEX energy (EUR/kg)"
            * "OPEX energy (%CAPEX)"

        given_sizing : int or float, optional
            The maximum capacity of the component, in kg.
            Relates to decision variable 'SE_S'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        name : str, optional
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max energy (kg)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        Notes
        -----
        1. This modelisation is equivalent to 2 OneVector storages with the constraint
           state_of_charge_1 + state_of_charge_2 = state_of_charge
           (i.e.: the discharge of one side leads to the charge of the other side).
        2. The instantaneous energetical state of charge of the storage depends on the product m(t) x Cp(t) x DT(t) where:

           * m(t) is the instantaneous mass state of charge of the storage (decision variable: E_S)
           * Cp(t) is the specific heat capacity of `stored_TVP`
           * DT(t) is the temperature difference between incoming and outcoming flows of `stored_TVP`

           Thus, in the general case, the storage can be charged when Cp(t) x DT(t) is low (requiring low charging power, decision variable: F_S)
           and discharged when Cp(t) x DT(t) is high; which has no physical meaning.
           For this reason, the Thermocline component must not be used with a ThermalVectorPair `stored_TVP` having a variable Cp(t) x DT(t) product.

        """
        if name is None:
            name = "TCline..{!r}".format(stored_TVP)
        super().__init__(properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self._add_used_element(stored_TVP, "element", _ThermalVectorPair)

    @property
    def stored_TVP(self):
        """
        Stored element.
        Storage is full when its entire content is the incoming vector of `stored_TVP`.

        """
        return self._element

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        if self.given_sizing is None:
            maxEnergyLowerBound = self._get_property("LB max energy (kg)")
            maxEnergyUpperBound = self._get_property("UB max energy (kg)")


            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                maxEnergyLowerBound * model_data.vars["N_units"][(hub, self)] <=
                model_data.vars["SE_S"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity LB", hub, self))]

            model_data.cts[hub][self] += [model_data.mdl.add_constraint(
                model_data.vars["SE_S"][(hub, self)] <=
                maxEnergyUpperBound * model_data.vars["N_units"][(hub, self)],
                ctname=NamesFormatter.fmt_light("Capacity UB", hub, self))]

        element = self._element

        charge_discharge_delay = self._get_property("Charge/discharge delay (h)")
        max_charging_discharging = model_data.vars["SE_S"][(hub, self)] / charge_discharge_delay
        CpDT_ = [DataAccessors.get2(element._CpDT, TS._t[ind_t], TS._t[ind_t + 1])
                 for ind_t in range(len(TS._t[:-1]))]
        model_data.cts[hub][self] += model_data.mdl.add_constraints((- max_charging_discharging * CpDT_[ind_t]
                                                                     <= model_data.vars["F_S"][
                                                                         (hub, element, self, TS._t[ind_t])]
                                                                     for ind_t in range(len(TS._t[:-1]))),
                                                                    names=NamesFormatter.fmt("Discharge UB", [hub],
                                                                                             [self], TS._t[:-1]))

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_S"][(hub, element, self, TS._t[ind_t])]
             <= max_charging_discharging * CpDT_[ind_t]
             for ind_t in range(len(TS._t[:-1]))),
            names=NamesFormatter.fmt("Charge UB", [hub], [self], TS._t[:-1]))

        energy_conservation = self._get_property("Energy conservation (/h)")
        charge_element = [0] * len(TS._t[:-1])
        for ind_t in range(len(TS._t[:-1])):
            # needed because if CpDT=0 at t,
            # then need to link E_S[t-1] and E_S[t+1] even if no F_S[t] is possible
            if CpDT_[ind_t] != 0:
                charge_element[ind_t] = model_data.vars["F_S"][(hub, element, self, TS._t[ind_t])] \
                                         * TS._dt[ind_t] * TS._step_value / CpDT_[ind_t]

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["E_S"][(hub, self, TS._t[ind_t + 1])]
             == model_data.vars["E_S"][(hub, self, TS._t[ind_t])]
             * energy_conservation ** (TS._dt[ind_t] * TS._step_value)
             + charge_element[ind_t]
             for ind_t in range(len(TS._t[:-2]))),
            names=NamesFormatter.fmt("SOC balance", [hub], [self], TS._t[:-2]))

        model_data.cts[hub][self] += [model_data.mdl.add_constraint((model_data.vars["E_S"][(hub, self, TS._t[0])]
                                                                     == model_data.vars["E_S"][(hub, self, TS._t[-2])]
                                                                     * energy_conservation ** (
                                                                             (TS._t[-1] - TS._t[-2]) * TS._step_value)    # TS._t[-1] - TS._t[-2] is TS._dt[-1]
                                                                     + charge_element[-1]),
                                                                    ctname=NamesFormatter.fmt_light(
                                                                        "SOC boundaries", hub, self, TS._t[-2]))]

    def _declare_KPIs(self, model_data, hub, TS):
        if self.eco_count:
            CAPEX = model_data.vars["SE_S"][(hub, self)] * self._get_property("CAPEX energy (EUR/kg)")
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
