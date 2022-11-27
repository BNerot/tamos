from copy import copy

from tamos.data_IO.data_IO import NamesFormatter, DataAccessors
from .elementIO_base import ElementIO


class Load(ElementIO):
    def __init__(self,
                 load,
                 element,
                 emissions=None,
                 element_cost=None,
                 exergy_count=True,
                 name=None):
        """
        Allows constrained element exchanges between the energy system and its environment.

        Load components are associated with the following exported decision variables:

        * X_EXT, binary.
          Whether the Load instance is used by the hub.
          To force the use of this instance by a hub, set components_assemblies at Hub or MILPModel level.
          For instance:

          >>> hub.components_assemblies = [(1, 1, heating_load)]

        * For all t, F_EXT(t), continuous, in kW.
          The power related to `element` entering the load (i.e. leaving the hub).

        Load components declare the following KPIs:

        * `ElementIO CO2`
           In kgEqCO2.
           Defines the "CO2" objective function. Related to the `emissions` attribute.
        * `ElementIO Exergy`
           In kWh.
           Defines the "Exergy" objective function. Related to the `exergy_factor` attribute of `element`.

        Parameters
        ----------
        load : int, float or numpy.ndarray
            The power pattern that constrains element exchanges.
            In kW.
            Same sign than the flow of 'element'.
        element : ElectricityVector, FuelVector, ThermalVector or ThermalVectorPair
        emissions : int, float or numpy.ndarray, optional
            Quantity of CO2 associated with a positive flow of `element`.
            In kgEqCO2/kWh.
            Usually negative.
            If None, emissions are not accounted for.
        element_cost : Cost, optional
            Cost associated with a positive flow of 'element'.
            If None, no costs are taken into account.
        exergy_count : bool, optional, default True
            Whether this instance contributes to the system "Exergy" KPI.
        name : str, optional

        Notes
        -----
        The Load class defines a multiplication operation to speed up the definition of several Load instances.
        Multiplication makes a copy of `load` attribute, so that mutable numpy.ndarray are made independent.
        For instance:

        >>> load_1 = Load(load=-4, element=element1, emissions=-0.2, exergy_count=False)
        >>> load_2 = 3 * load_1
        >>> load_2.load
         -12
        >>> load_2.element is element1
         True


        Examples
        --------
        >>> Load(load=10, element=electricity)

        A constant electrical power of 10 kW will leave the hub interface.

        >>> thermal_source = fetch_TVP(in_TV=in_TV, out_TV=out_TV)
        >>> load_1 = Load(load=-5, element=thermal_source)
        >>> load_2 = Load(load=5, element=~thermal_source)

        In both cases of load_1 and load_2, a constant power of 5 kW is exchanged between the hub interface and the Load component,
        with `out_TV` entering the hub interface (i.e. leaving the Load component).

        """
        if name is None:
            name = f"Load..{element}"
        super().__init__(element=element,
                         emissions=emissions,
                         element_cost=element_cost,
                         exergy_count=exergy_count,
                         name=name)
        self.load = load

    @property
    def load(self):
        """
        The power pattern that constrains element exchanges.
        In kW.
        Same sign than the flow of 'element'.
        int, float or numpy.ndarray

        """
        return self._load

    @load.setter
    def load(self, load):
        DataAccessors.type_checker(load, self, "load", "numeric")
        self._load = load

    def _declare_constraints(self, model_data, hub, TS):
        super()._declare_constraints(model_data, hub, TS)
        cts = [0] * len(TS._t[:-1])
        for ind_t in range(len(TS._t[:-1])):
            t1, t2 = TS._t[ind_t], TS._t[ind_t + 1]
            load = DataAccessors.get2(self.load, t1, t2)
            name = NamesFormatter.fmt_light(f"Assignation", hub, self._element, self, t1)
            cts[ind_t] = model_data.mdl.indicator_constraint(model_data.vars["X_EXT"][(hub, self)],
                                                             model_data.vars["F_EXT"][(hub, self, t1)] == load,
                                                             name=name)

        model_data.cts[hub][self] += model_data.mdl.add_indicator_constraints(cts)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        new_load = Load(
                 load=copy(self.load) * other, # copy to prevent sharing the same numpy array
                                               # (another solution is to overload __deepcopy__)
                 element=self.element,
                 emissions=self.emissions,
                 element_cost=self.element_cost,
                 exergy_count=self.exergy_count,
                 name=self.name)
        return new_load

    def __rmul__(self, other):
        return self.__mul__(other)