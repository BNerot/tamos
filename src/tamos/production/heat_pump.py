from numpy import where, ndim
from pandas import DataFrame
from sympy import ln, exp, var, Piecewise, lambdify


from ..data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair, ThermalVector, ElectricityVector
from .production_base import Production


class HPChillerSHCHP(Production):
    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties,
                 given_sizing=None,
                 name=None, eco_count=True,
                 units_number_ub=1, units_number_lb=1):
        if name is None:
            if isinstance(self, CompHP):
                name = "Comp-HP..{!r}..{!r}".format(energy_source, energy_sink)
            else:
                assert isinstance(self, AbsHP)
                name = "Abs-HP..{!r}..{!r}".format(energy_source, energy_sink)
        super().__init__(properties, name, given_sizing, units_number_ub, units_number_lb, eco_count)
        self._add_used_element(energy_source, "energy_source", (ThermalVector, _ThermalVectorPair))
        self._add_used_element(energy_sink, "energy_sink", (ThermalVector, _ThermalVectorPair))
        if isinstance(energy_source, _ThermalVectorPair):
            assert energy_source.is_cooled, f"{self}: 'energy_source' must be cooled down. consider using ~({energy_source})."
        if isinstance(energy_sink, _ThermalVectorPair):
            assert energy_sink.is_warmed, f"{self}: 'energy_sink' must be warmed up. consider using ~({energy_sink})."

        self.ref_production = energy_sink

    @property
    def energy_source(self):
        """
        Element that gives thermal energy.
        Must be cooled down if ThermalVectorPair.

        """
        return self._energy_source

    @property
    def energy_sink(self):
        """
        Element that receives thermal energy.
        Must be warmed up if ThermalVectorPair.                -------

        """
        return self._energy_sink

    @property
    def energy_drive(self):
        """
        Element enabling the pressurization of the heat pump refrigerant.

        """
        return self._energy_drive

    @property
    def ref_production(self):
        """
        Defines whether `energy_sink` or `energy_source` is the reference production.
        Either `energy_sink` or `energy_source`.

        The reference production defines decision variable 'Q_P' and thus 'SP_P',
        which defines the cost of the component.
        If costs properties are typical values for a heat pump use, set `HP.ref_production=HP.energy_sink`.
        If the costs properties are given for a chiller use, set `HP.ref_production=HP.energy_source`.

        """
        return self._ref_production

    @ref_production.setter
    def ref_production(self, ref_production):
        assert ref_production in [self.energy_sink,
                                  self.energy_source], \
            f"{self}: 'ref_production' must be one of [{self.energy_sink}, {self.energy_source}]."
        self._ref_production = ref_production

    @property
    def efficiency(self):
        """
        Defines explicitly the efficiency of the heat pump, i.e. the coefficient of performance (COP).

        The COP is a heating COP, i.e. it is used according to the two following constraints:
        1. energy_from_source(t) + energy_from_drive(t) = energy_to_sink(t)
        2. energy_from_sink(t) = energy_from_drive(t) * COP(t)

        If called, replaces the definition of the efficiency using `set_efficiency_model` (default).
        int, float or numpy.ndarray

        """
        return self._efficiency

    @efficiency.setter
    def efficiency(self, efficiency):
        DataAccessors.type_checker(efficiency, self, "efficiency", "numeric")
        self._efficiency = efficiency

    def set_efficiency_model(self, source_pinch, sink_pinch, as_HEx=True, **model_kwargs):
        """
        Defines the efficiency of the heat pump, i.e. the coefficient of performance (COP).

        The COP is a heating COP, i.e. it is used according to the two following constraints:
        1. energy_from_source(t) + energy_from_drive(t) = energy_to_sink(t)
        2. energy_from_sink(t) = energy_from_drive(t) * COP(t)

        Parameters
        ----------
        source_pinch : int, float or numpy.ndarray
            Temperature difference between `energy_source` and the refrigerant fluid of the heat pump (evaporator side).
            If `energy_source` is a ThermalVectorPair, the relevant temperature is the one of the cold vector (outcoming).
        sink_pinch : int, float or numpy.ndarray
            Temperature difference between `energy_sink` and the refrigerant fluid of the heat pump (condenser side).
            If `energy_sink` is a ThermalVectorPair, the relevant temperature is the one of the warm vector (outcoming).
        as_HEx : bool, optional, default True
            Describes the behavior of the component when the temperature of the heat source exceeds the one of the heat sink.

            * If True, the heat pump behaves similarly as a heat exchanger since its COP is very high.
              This high COP is calculated by defining equal temperatures for energy source and sink:
              new_source_temperature(t) = new_sink_temperature(t) = (source_temperature(t)+sink_temperature(t))/2
            * If False, COP model is applied as of, which could lead to non physical results.

        model_kwargs : keyword arguments passed to the COP calculation function.

            * For CompHP components, these can be:

              * 'fluid': {'ammonia', 'isobutane'}, default 'ammonia'
                Refrigerant fluid of the heat pump.
              * 'eta_is', float, default 0.75
                Isentropic efficiency of the compression.
              * 'f_Q': float, default 0.2
                Compressor heat loss ratio.

            * For AbsHP components, these can be:

              * 'couple': {'NH3/H2O', 'H2O/LiBr'}, default 'H2O/LiBr'
                Refrigerant and absorbent fluids.

        Notes
        -----
        See class `COPModels` from tamos.production.

        """
        DataAccessors.type_checker(source_pinch, self, "source_pinch", "numeric")
        DataAccessors.type_checker(sink_pinch, self, "sink_pinch", "numeric")
        DataAccessors.type_checker(as_HEx, self, "as_HEx", bool)
        DT_P_source = source_pinch  # pinch between evaporator and fluid at secondary side
        if isinstance(self._energy_source, _ThermalVectorPair):
            T_source_o = self._energy_source._out_TV.temperature
            DT_source = self._energy_source._DT
        else:  # ThermalVector                #for instance, energy_source = infinite lake
            T_source_o = self._energy_source.temperature
            DT_source = 0.01

        DT_P_sink = sink_pinch  # pinch between condenser and fluid at secondary side
        if isinstance(self._energy_sink, _ThermalVectorPair):
            T_sink_o = self._energy_sink._out_TV.temperature
            DT_sink = self._energy_sink._DT
        else:  # ThermalVector                #for instance, energy_sink = infinite lake
            T_sink_o = self._energy_sink.temperature
            DT_sink = 0.01

        if as_HEx:
            T_sink_o, T_source_o = where(T_sink_o < T_source_o, (T_sink_o + T_source_o) / 2, T_sink_o), \
                                   where(T_sink_o < T_source_o, (T_sink_o + T_source_o) / 2, T_source_o)
            # warning: double assignation is dealt with, do not change

        self._apply_COP_func(T_sink_o, DT_sink, T_source_o, DT_source, DT_P_sink, DT_P_source, **model_kwargs)
        self.efficiency = where(self.efficiency < 1, 0, self.efficiency)
        if ndim(self.efficiency) == 0:  # if ndarray of dim 0 can't be handled by DataAccessors.get methods
            self.efficiency = float(self.efficiency)

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        # energy sink balance, depends on source type
        if isinstance(self._energy_sink, _ThermalVectorPair):
            sign_sink = 1
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_lb(0) for t in TS._t[:-1]]
        else:  # ThermalVector                #for instance, energy_sink = infinite lake
            sign_sink = -1
            [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_ub(0) for t in TS._t[:-1]]
        model_data.cts[hub][self] += model_data.mdl.add_constraints((model_data.mdl.eq_constraint(
            sign_sink * model_data.vars["F_P"][(hub, self._energy_sink, self, TS._t[ind_t])],
            model_data.vars["F_P"][(hub, self._energy_drive, self, TS._t[ind_t])]
            * DataAccessors.get2(self.efficiency,
                                 TS._t[ind_t],
                                 TS._t[ind_t + 1]))
            for ind_t in range(len(TS._t[:-1]))),
            names=NamesFormatter.fmt("Power balance 1", [hub],
                                     [self],
                                     TS._t[:-1]))
        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_P"][(hub, self._energy_drive, self, t)]
             == sign_sink * model_data.vars["F_P"][(hub, self._energy_sink, self, t)]
             - model_data.vars["F_P"][(hub, self._energy_source, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Power balance 2", [hub], [self],
                                     TS._t[:-1]))
        if self.ref_production == self.energy_sink:
            model_data.cts[hub][self] += model_data.mdl.add_constraints((model_data.vars["Q_P"][
                                                                             (hub, self, t)] == sign_sink *
                                                                         model_data.vars["F_P"][
                                                                             (hub, self._energy_sink, self, t)]
                                                                         for t in TS._t[:-1]),
                                                                        names=NamesFormatter.fmt("Ref production",
                                                                                                 [hub], [self],
                                                                                                 TS._t[:-1]))
        else:
            model_data.cts[hub][self] += model_data.mdl.add_constraints(
                (model_data.vars["Q_P"][(hub, self, t)]
                 == model_data.vars["F_P"][(hub, self._energy_source, self, t)]
                 for t in TS._t[:-1]),
                names=NamesFormatter.fmt("Ref production", [hub], [self],
                                         TS._t[:-1]))

        [model_data.vars["F_P"][(hub, self._energy_drive, self, t)].set_lb(0) for t in TS._t[:-1]]
        [model_data.vars["F_P"][(hub, self._energy_source, self, t)].set_lb(0) for t in TS._t[:-1]]


class CompHP(HPChillerSHCHP):
    def __init__(self,
                 energy_drive,
                 energy_source,
                 energy_sink,
                 properties,
                 given_sizing=None,
                 name=None, eco_count=True,
                 units_number_ub=1, units_number_lb=1):
        """
        CompHP components describe vapor-compression heat pumps.
        The COP model is an adaptation of (Jensen et al., 2018) [1]_.


        This component declares the following exported decision variables:

        * X_P, binary.
          Whether the component is used by the hub.
        * SP_P, continuous, in kW.
          The maximum capacity of the component. Defines the investment costs.
        * For all t, for all element e, F_P(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).
        * For all t, Q_P(t), continuous, in kW.
          The reference power related to the component. Defines the variable cost.
          This power is a lower bound of SP_P.
          There exists one element e such that Q_P(t) = F_P(e, t) or Q_P(t) = - F_P(e, t).
          For this component, e is either `energy_sink` or `energy_source` depending on the `ref_production` attribute.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_drive : ElectricityVector
            Electricity consumed by the machine.
        energy_source : ThermalVectorPair, ThermalVector
            Element that gives thermal energy.
            Must be cooled down if ThermalVectorPair.
        energy_sink : ThermalVectorPair, ThermalVector
            Element that receives thermal energy.
            Must be warmed up if ThermalVectorPair.
        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/kW)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        given_sizing : int or float, optional
            The maximum capacity of the component, in kW.
            Relates to decision variable 'SP_P'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        name : str, optional
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.

        References
        ----------
        .. [1] JENSEN J. K, OMMEN T, REINHOLDT L, Et Al. Heat pump COP, part 2: generalized COP estimation of heat pump processes.
               2018. https://doi.org/10.18462/IIR.GL.2018.1386.

        """
        super().__init__(energy_source=energy_source,
                         energy_sink=energy_sink,
                         properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self._add_used_element(energy_drive, "energy_drive", ElectricityVector)
        self.set_efficiency_model(3, 3, True)

    def _apply_COP_func(self, T_sink_o, DT_sink, T_source_o, DT_source, DT_P_sink, DT_P_source, **model_kwargs):
        COP_comp = COPModels.compression(**model_kwargs)
        self.efficiency = COP_comp(T_sink_o, DT_sink, T_source_o, DT_source, DT_P_sink, DT_P_source)


class AbsHP(HPChillerSHCHP):
    def __init__(self,
                 energy_drive,
                 energy_source,
                 energy_sink,
                 electricity_aux,
                 properties,
                 given_sizing=None, pump_consumption=0.026, name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        AbsHP components describe absorption heat pumps.
        The COP model is an adaptation of (Boudéhenn et al., 2016) [2]_.


        This component declares the following exported decision variables:

        * X_P, binary.
          Whether the component is used by the hub.
        * SP_P, continuous, in kW.
          The maximum capacity of the component. Defines the investment costs.
        * For all t, for all element e, F_P(e, t), continuous, in kW.
          The power related to element e entering the component (i.e. leaving the hub interface).
        * For all t, Q_P(t), continuous, in kW.
          The reference power related to the component. Defines the variable cost.
          This power is a lower bound of SP_P.
          There exists one element e such that Q_P(t) = F_P(e, t) or Q_P(t) = - F_P(e, t).
          For this component, e is either `energy_sink` or `energy_source` depending on the `ref_production` attribute.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_drive : ThermalVectorPair
            Thermal flow that is cooled down for the desorption process.
        energy_source : ThermalVectorPair, ThermalVector
            Element that gives thermal energy.
            Must be cooled down if ThermalVectorPair.
        energy_sink : ThermalVectorPair, ThermalVector
            Element that receives thermal energy.
            Must be warmed up if ThermalVectorPair.
        electricity_aux : ElectricityVector
            Electricity consumed by the machine.
        properties : dict {str: int | float}
            Techno-economic properties of the component.
            The `properties` attribute must include the following keys:

            * "LB max output power (kW)"
            * "UB max output power (kW)"
            * "CAPEX (EUR/kW)"
            * "OPEX (%CAPEX)"
            * "Variable OPEX (EUR/MWh)"

        given_sizing : int or float, optional
            The maximum capacity of the component, in kW.
            Relates to decision variable 'SP_P'.
            If specified, only the operation of this component is performed by the MILP solver.
            If let unknown, both sizing and operation are performed.
        pump_consumption : float, optional, default 0.026
            Electricity consumption of the heat pump, in proportion of the power extracted from `energy_source`.
        name : str, optional
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.

        References
        ----------
        .. [2] Boudéhenn F, Bonnot S, Demasles H, Lefrançois F, Perier-Muzet M, Triché D.
               Development and Performances Overview of Ammonia-water Absorption Chillers with Cooling Capacities from 5 to 100 kW.
               Energy Procedia 2016;91:707–16. https://doi.org/10.1016/j.egypro.2016.06.234.

        """
        super().__init__(energy_source=energy_source,
                         energy_sink=energy_sink,
                         properties=properties,
                         given_sizing=given_sizing,
                         name=name,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)

        self._add_used_element(energy_drive, "energy_drive", (ThermalVector, _ThermalVectorPair))
        if isinstance(energy_drive, _ThermalVectorPair):
            assert energy_drive.is_cooled, f"{self}: 'energy_drive' must be cooled down. consider using ~({energy_drive})."
        self._add_used_element(electricity_aux, "electricity_aux", ElectricityVector)
        self.pump_consumption = pump_consumption
        if name is None:
            self.name = "Abs..{!r}-HP..{!r}..{!r}".format(energy_drive, energy_source, energy_sink)
        self.set_efficiency_model(3, 3, True)

    @property
    def pump_consumption(self):
        return self._pump_consumption

    @pump_consumption.setter
    def pump_consumption(self, pump_consumption):
        DataAccessors.type_checker(pump_consumption, self, "pump_consumption", "numeric")
        self._pump_consumption = pump_consumption

    def _apply_COP_func(self, T_sink_o, DT_sink, T_source_o, DT_source, DT_P_sink, DT_P_source, **model_kwargs):
        if isinstance(self._energy_drive, _ThermalVectorPair):
            T_drive_i = self._energy_drive._in_TV.temperature
        else:
            T_drive_i = self._energy_drive.temperature
        COP_abs = COPModels.absorption(**model_kwargs)
        self.efficiency = COP_abs(T_sink_o, DT_sink, T_source_o, DT_source, DT_P_sink, DT_P_source,
                                  T_drive_i)

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.mdl.eq_constraint(model_data.vars["F_P"][(hub, self._electricity_aux, self, TS._t[ind_t])],
                                          DataAccessors.get2(self.pump_consumption, TS._t[ind_t], TS._t[ind_t + 1])

                                          * model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])])
             for ind_t in range(len(TS._t[:-1]))),
            names=NamesFormatter.fmt("pump_consumption", [hub], [self], TS._t[:-1]))


class COPModels:
    # H: heating
    # C: cooling
    # ent: entropic
    # P: pinch
    # is: isentropic
    # I/O: incoming/outcoming

    @staticmethod
    def absorption(couple = "H2O/LiBr"):
        """
        Computes the expression of the COP model for absorption heat pumps.

        This model is an adaptation of (Boudéhenn et al., 2016) [2]_.

        Parameters
        ----------
        couple : {'NH3/H2O', 'H2O/LiBr'}, default 'H2O/LiBr'
           Refrigerant and absorbent fluids.

        Returns
        -------
        A sympy callable returning the COP. Its arguments are:

           * T_H_O: temperature of the outcoming flow of energy_sink, in Kelvins (K)
           * DT_H: difference of temperature between incoming and outcoming flows of energy_sink, in Kelvins (K)
           * T_C_O: temperature of the outcoming flow of energy_source, in Kelvins (K)
           * DT_C: difference of temperature between incoming and outcoming flows of energy_source, in Kelvins (K)
           * DT_P_H: difference of temperature between the refrigerant fluid and energy_sink, in Kelvins (K)
           * DT_P_C: difference of temperature between the refrigerant fluid and energy_source, in Kelvins (K)
           * T_G_I: temperature of the incoming flow of energy_drive, in Kelvins (K)

        Notes
        -----
        The COP is a heating COP, i.e. it is used according to the two following constraints:
        1. energy_from_source(t) + energy_from_drive(t) = energy_to_sink(t)
        2. energy_from_sink(t) = energy_from_drive(t) * COP(t)

        References
        ----------
        .. [2] Boudéhenn F, Bonnot S, Demasles H, Lefrançois F, Perier-Muzet M, Triché D.
               Development and Performances Overview of Ammonia-water Absorption Chillers with Cooling Capacities from 5 to 100 kW.
               Energy Procedia 2016;91:707–16. https://doi.org/10.1016/j.egypro.2016.06.234.


        """
        var("T_H_O DT_H "
            "T_C_O DT_C DT_P_H DT_P_C T_G_I C")
        T_H_I = T_H_O - DT_H
        T_C_I = T_C_O + DT_C
        T_H_ent = (T_H_O - T_H_I) / ln(T_H_O / T_H_I)
        T_C_ent = (T_C_O - T_C_I) / ln(T_C_O / T_C_I)
        T_COND = T_H_ent + DT_P_H
        T_EV = T_C_ent - DT_P_C
        A1, A2 = -1.26844, -35.60648
        B1, B2 = 0.97213, 0.21713
        C_ = {"NH3/H2O": 0.85, "H2O/LiBr": 1.1}
        COP_Carnot = (T_EV / T_G_I) * (T_G_I - T_COND) / (T_COND - T_EV)

        EER = C * (Piecewise((A1 * exp(-COP_Carnot / B1) + A2 * exp(-COP_Carnot / B2), T_H_O > 280.25), (0, True))
                   + 0.75)
        COP = 1 + EER
        COP = COP.subs({C: C_[couple]})

        COP_abs = lambdify([T_H_O, DT_H, T_C_O, DT_C, DT_P_H, DT_P_C, T_G_I], COP)
        return COP_abs

    @staticmethod
    def compression(fluid = "ammonia",
                    eta_is=0.75,
                    f_Q=0.2):
        """
        Computes the expression of the COP model for compression heat pumps.

        This model is an adaptation of (Jensen et al., 2018) [1]_.

        Parameters
        ----------
        fluid : {'ammonia', 'isobutane'}, optional, default 'ammonia'
            Refrigerant fluid of the heat pump.
        eta_is: float, default 0.75
            Isentropic efficiency of the compression.
        f_Q: float, default 0.2
            Compressor heat loss ratio.

        Returns
        -------
        A sympy callable returning the COP. Its arguments are:

           * T_H_O: temperature of the outcoming flow of energy_sink, in Kelvins (K)
           * DT_H: difference of temperature between incoming and outcoming flows of energy_sink, in Kelvins (K)
           * T_C_O: temperature of the outcoming flow of energy_source, in Kelvins (K)
           * DT_C: difference of temperature between incoming and outcoming flows of energy_source, in Kelvins (K)
           * DT_P_H: difference of temperature between the refrigerant fluid and energy_sink, in Kelvins (K)
           * DT_P_C: difference of temperature between the refrigerant fluid and energy_source, in Kelvins (K)

        Notes
        -----
        The COP is a heating COP, i.e. it is used according to the two following constraints:
        1. energy_from_source(t) + energy_from_drive(t) = energy_to_sink(t)
        2. energy_from_sink(t) = energy_from_drive(t) * COP(t)


        References
        ----------
        .. [1] JENSEN J. K, OMMEN T, REINHOLDT L, Et Al. Heat pump COP, part 2: generalized COP estimation of heat pump processes.
               2018. https://doi.org/10.18462/IIR.GL.2018.1386.

        """
        var("T_H_O DT_H T_C_O DT_C DT_P_H DT_P_C f_Q_ eta_is_ a1 a2 b1 b2 c1 c2")
        parameters = {"ammonia": {a1: 0.2, b1: 0.2, c1: 0.016,
                                  a2: 0.0014, b2: -0.0015, c2: 0.039},
                      "isobutane": {a1: -0.0011, b1: 0.30, c1: 2.4,
                                    a2: 0.0035, b2: -0.0033, c2: 0.053}}
        T_H_I = T_H_O - DT_H
        T_C_I = T_C_O + DT_C
        T_H_ent = (T_H_O - T_H_I) / ln(T_H_O / T_H_I)
        T_C_ent = (T_C_O - T_C_I) / ln(T_C_O / T_C_I)
        DT_lift_ent = T_H_ent - T_C_ent
        COP_Lorenz = T_H_ent / DT_lift_ent
        DT_P_H_ent = DT_P_H
        DT_P_C_ent = DT_P_C
        DT_r_C_ent = 0.5 * DT_C
        DT_r_H_ent = a1 * (T_H_O - T_C_O + DT_P_H + DT_P_C) + b1 * DT_H + c1
        R_is = a2 * (T_H_O - T_C_O + DT_P_H + DT_P_C) + b2 * DT_H + c2

        # eq. 10 in (Jensen, 2018)
        # warning: error in eq 10 (last term) corrected here after
        efficiency = ((1 + (DT_r_H_ent + DT_P_H_ent) / T_H_ent) /
                      (1 + (DT_r_H_ent + DT_r_C_ent + DT_P_H_ent + DT_P_C_ent) / DT_lift_ent)) \
                     * eta_is_ * (1 - R_is) + (1 - eta_is_ - f_Q_) * (DT_lift_ent / T_H_ent)
        COP = efficiency * COP_Lorenz
        COP = COP.subs({eta_is_: eta_is, f_Q_: f_Q}).subs(parameters[fluid])
        COP_comp = lambdify([T_H_O, DT_H, T_C_O, DT_C, DT_P_H, DT_P_C], COP)  # to sympy function, numpy
        return COP_comp



    @staticmethod
    def plot(range_T_H_O=range(5+273, 71+273, 1),
             range_T_C_O=range(0+273, 26+273, 5),
             range_T_G_I=range(90+273, 130+273, 10),
             celsius=False):
        """
        Plots typical values returned by the absorption and compression COP models.

        Parameters
        ----------
        range_T_H_O : list-like, optional, default range(5+273, 71+273, 1)
            Temperatures of the outcoming flow of energy_sink, in Kelvins (K).
        range_T_C_O : list-like, optional, default range(0+273, 26+273, 5)
            Temperatures of the outcoming flow of energy_source, in Kelvins (K).
        range_T_G_I : list-like, optional, default range(90+273, 130+273, 10)
            Temperatures of the incoming flow of energy_drive, in Kelvins (K).
            Relevant for absorption COP model.
        celsius: bool, optional, default False
            If True, arguments `range_T_H_O`, `range_T_C_O` and `range_T_G_I` must describe temperatures in Celsius (°C).

        Returns
        -------
        figs: list of plotly.graph_objs._figure.Figure

        Notes
        -----
        Figures are displayed on the local host browser using the plotly package.

        """
        try:
            from plotly.express import line
            from plotly.io import renderers
            renderers.default = "browser"
        except ModuleNotFoundError:
            print(f"COPModels: the 'plotly' package is required to plot the COP values.")

        DT_H = 0.001
        DT_C = 0.001
        DT_P_H = 3
        DT_P_C = 3
        data = [(T_H_O, DT_H, T_C_O, DT_C, DT_P_H, DT_P_C, T_G_I)
                for T_H_O in range_T_H_O
                for T_C_O in range_T_C_O
                for T_G_I in range_T_G_I
                if T_H_O >= T_C_O
                ]
        columns = ["T_H_O", "DT_H", "T_C_O", "DT_C", "DT_P_H", "DT_P_C", "T_G_I"]
        df = DataFrame(data, columns=columns)
        for COP_func, suffix, col_attr in zip([COPModels.absorption(), COPModels.compression()],
                                              ["abs", "comp"],
                                              [None, -1]):
            if celsius:
                df[f"COP_{suffix}"] = COP_func(*df[columns[:col_attr]]
                                               .apply(lambda col: col + 273 if "DT_" not in col.name else col)
                                               .values.transpose())
            else:
                df[f"COP_{suffix}"] = COP_func(*df[columns[:col_attr]]
                                               .values.transpose())

            # df[f"EER_{suffix}"] = df[f"COP_{suffix}"] - 1
            # df[f"EER_{suffix}"] = df[f"EER_{suffix}"].where((df[f"EER_{suffix}"]>0)&(df[f"EER_{suffix}"]<10), nan)
        figs = []
        for y_, color in zip(["COP_abs", "COP_comp"],
                              ["T_G_I", None]):
            title = f"{y_} | DT_H = {DT_H} | DT_C = {DT_H} | DT_P_H = {DT_P_H} | DT_P_C = {DT_P_C}"
            fig = line(df, x="T_H_O", y=y_, facet_col="T_C_O", facet_col_wrap=2, color=color, title=title)
            fig.show()
            figs.append(fig)

        return figs
