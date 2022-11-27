from numpy import where, maximum, ndim

from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from tamos.element import _ThermalVectorPair, FuelVector, ElectricityVector
from .production_base import Production


class Boiler(Production):

    def __init__(self,
                 energy_sink,
                 properties,
                 given_sizing,
                 name,
                 units_number_ub,
                 units_number_lb,
                 eco_count):
        super().__init__(
            properties=properties,
            name=name,
            given_sizing=given_sizing,
            units_number_ub=units_number_ub,
            units_number_lb=units_number_lb,
            eco_count=eco_count)

        self._add_used_element(energy_sink, "energy_sink", _ThermalVectorPair)
        assert energy_sink.is_warmed, f"{self}: 'energy_sink' must be warmed up. consider using ~({energy_sink})."

    @property
    def energy_source(self):
        return self._energy_source

    @property
    def energy_sink(self):
        """
        Thermal flow that is warmed up by the boiler.

        """
        return self._energy_sink

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["F_P"][(hub, self._energy_sink, self, TS._t[ind_t])]
             == model_data.vars["F_P"][(hub, self._energy_source, self, TS._t[ind_t])]
             * DataAccessors.get2(self.efficiency, TS._t[ind_t], TS._t[ind_t + 1])
             for ind_t in range(len(TS._t[:-1]))),
            names=NamesFormatter.fmt("Power balance", [hub],
                                     [self], TS._t[:-1]))

        model_data.cts[hub][self] += model_data.mdl.add_constraints(
            (model_data.vars["Q_P"][(hub, self, t)] == model_data.vars["F_P"][(hub, self._energy_sink, self, t)]
             for t in TS._t[:-1]),
            names=NamesFormatter.fmt("Ref production", [hub], [self],
                                     TS._t[:-1]))

        [model_data.vars["F_P"][(hub, self._energy_source, self, t)].set_lb(0) for t in TS._t[:-1]]
        [model_data.vars["F_P"][(hub, self._energy_sink, self, t)].set_lb(0) for t in TS._t[:-1]]

    @property
    def efficiency(self):
        """
        Defines explicitly the efficiency of the conversion of `energy_source` to `energy_sink`.
        If called, replaces the definition of the efficiency using `set_efficiency_model` (default).
        int, float or numpy.ndarray

        """
        return self._efficiency

    @efficiency.setter
    def efficiency(self, efficiency):
        DataAccessors.type_checker(efficiency, self, "efficiency", "numeric")
        self._efficiency = efficiency

    def set_efficiency_model(self, efficiency_function, pinch):
        """
        Defines the efficiency of the conversion of `energy_source` to `energy_sink` using
        a function of the cold temperature of `energy_sink`.

        Parameters
        ----------
        efficiency_function : callable f(T)
            T is the temperature of the cold vector of `energy_sink`, in Kelvins (K).
        pinch : int, float or numpy.ndarray
            Temperature difference between the flue gases of the boiler and the cold vector of `energy_sink`, in Kelvins (K).

        Notes
        -----
        By default, set_efficiency_model is called with the `default_efficiency` attribute of this instance and pinch = 2.

        """
        DataAccessors.type_checker(pinch, self, "pinch", "numeric")

        T = self._energy_sink._in_TV.temperature + pinch
        efficiency = efficiency_function(T)
        if ndim(efficiency) == 0:
            # if T is float then self._efficiency is ndarray of dim 0
            # which can't be handled by DataAccessors.get methods
            efficiency = float(efficiency)
        self.efficiency = efficiency


class FuelBoiler(Boiler):
    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties,
                 pinch,
                 efficiency_function,
                 given_sizing=None,
                 name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        super().__init__(energy_sink=energy_sink,
                         properties=properties,
                         name=name,
                         given_sizing=given_sizing,
                         units_number_ub=units_number_ub,
                         units_number_lb=units_number_lb,
                         eco_count=eco_count)
        self._add_used_element(energy_source, "energy_source", FuelVector)
        self.set_efficiency_model(efficiency_function, pinch)


class GasBoiler(FuelBoiler):

    @staticmethod
    def _quadratic_efficiency(T): return (-(T ** 2) / 183.7 + T / 7.8 + 108.1) / 100  # in °C

    @staticmethod
    def _linear_efficiency(T): return (-T / 25.6 + 100.3) / 100  # in °C

    @staticmethod
    def default_efficiency(T): return where(T > 56 + 273,
                                            GasBoiler._linear_efficiency(T - 273),
                                            GasBoiler._quadratic_efficiency(maximum(T - 273, 20)))

    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties,
                 given_sizing=None,
                 name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        GasBoiler components produce heat given an energy efficiency of typical gas boilers.
        This efficiency takes into account the condensing property of flue gases.

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
          For this component, e is `energy_sink`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_source : FuelVector
            Natural gas that is consumed.
        energy_sink : ThermalVectorPair
            Thermal flow that is warmed up by the boiler.
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
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        """
        if name is None:
            name = "Gas-Boil..{!r}".format(energy_sink)
        super().__init__(
            energy_source=energy_source,
            energy_sink=energy_sink,
            properties=properties,
            pinch=2,
            efficiency_function=GasBoiler.default_efficiency,
            given_sizing=given_sizing,
            name=name,
            units_number_ub=units_number_ub,
            units_number_lb=units_number_lb,
            eco_count=eco_count)


class BiomassBoiler(FuelBoiler):
    @staticmethod
    def _quadratic_efficiency(T): return (-(T ** 2) / 66.1 + T / 1.22 + 91.8) / 100

    @staticmethod
    def _linear_efficiency(T): return 76.5 / 100

    @staticmethod
    def default_efficiency(T): return where(T > 68 + 273,
                                            BiomassBoiler._linear_efficiency(T - 273),
                                            BiomassBoiler._quadratic_efficiency(maximum(T - 273, 20)))

    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties, given_sizing=None,
                 name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        BiomassBoiler components produce heat given an energy efficiency of typical biomass boilers.
        This efficiency takes into account the condensing property of flue gases.

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
          For this component, e is `energy_sink`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_source : FuelVector
            Biomass that is consumed.
        energy_sink : ThermalVectorPair
            Thermal flow that is warmed up by the boiler.
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
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        """
        if name is None:
            name = "Biom-Boil..{!r}".format(energy_sink)
        super().__init__(
            energy_source=energy_source,
            energy_sink=energy_sink,
            properties=properties,
            pinch=2,
            efficiency_function=BiomassBoiler.default_efficiency,
            given_sizing=given_sizing,
            name=name,
            units_number_ub=units_number_ub,
            units_number_lb=units_number_lb,
            eco_count=eco_count)


class ElectricHeater(Boiler):
    def __init__(self,
                 energy_source,
                 energy_sink,
                 properties,
                 given_sizing=None,
                 name=None,
                 units_number_ub=1, units_number_lb=1, eco_count=True):
        """
        ElectricHeater components produce heat following a unit efficiency.

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
          For this component, e is `energy_sink`.

        This component declares the following KPIs:

        * `COST_production`
          In euros.
          Contributes to the "Eco" objective function.


        Parameters
        ----------
        energy_source : ElectricityVector
        energy_sink : ThermalVectorPair
            Thermal flow that is warmed up by the boiler.
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
        units_number_lb, units_number_ub : int, optional, default 1
            The lower bound (upper bound) of the number of real components that this instance aims to stand for.
            Setting `units_number_lb` (`units_number_ub`) has a meaning if "LB max output power (kW)" property is
            different from 0.
        eco_count : bool, optional, default True
            Whether this instance contributes to the system "Eco" KPI.

        """
        if name is None:
            name = "Elec-Boil..{!r}".format(energy_sink)
        super().__init__(
            energy_sink=energy_sink,
            properties=properties,
            given_sizing=given_sizing,
            name=name,
            units_number_ub=units_number_ub,
            units_number_lb=units_number_lb,
            eco_count=eco_count)
        self.efficiency = 1
        self._add_used_element(energy_source, "energy_source", ElectricityVector)

    def set_efficiency_model(self, efficiency_function, pinch):
        print(
            f"{self}: a call to this method is not relevant for ElectricHeater. Please set the 'efficiency' attribute.")
        pass
