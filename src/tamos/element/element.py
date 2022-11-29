from itertools import chain

from numpy import log, all, any

from tamos.component import MetaComponent
from ..data_IO.data_IO import DataAccessors


class Element(MetaComponent):

    _dead_state_temperature = 25 + 273
    @staticmethod
    def _get_dead_state_temperature():
        """
        The dead state temperature is the reference temperature automatically used in calculation of attribute `exergy_factor` of
        ThermalVector and ThermalVectorPair instances.
        It can be modified using the `set_dead_state_temperature` function of tamos.element.

        Returns
        -------
        The dead state temperature, in Kelvins (K).

        """
        return Element._dead_state_temperature

    @staticmethod
    def _set_dead_state_temperature(dead_state_temperature):
        """
        Assign a new value to the dead state temperature, which is common to all elements.
        Already defined elements are not affected by this change.

        Parameters
        ----------
        dead_state_temperature : int, float or numpy.ndarray

        """
        DataAccessors.type_checker(dead_state_temperature, Element, "dead_state_temperature", "numeric")
        print(f"Warning: The change in 'dead_state_temperature' has no effect on already defined elements."
              f" It will impact all future elements.")
        Element._dead_state_temperature = dead_state_temperature

    @staticmethod
    def _decompose_elements(elements: "List of Element objets"):
        """
        Returns
        -------
        A a set of all vectors used in `elements`.

        """
        return set(chain(*[element.get_vectors() for element in elements]))

    def get_vectors(self):
        """
        Returns this instance.

        """
        # for compatibility with ThermalVectorPair
        return [self]

    @property
    def exergy_factor(self):
        """
        Quantity of exergy associated with a 1 kW power flow of this element.
        Theoretically positive, usually smaller than 1. In kW/kW.

        int, float or numpy.ndarray
        0 <= exergy_factor

        Notes
        -----
        FuelVector instances are exchanged between components considering their LHV value.

        """
        return self._exergy_factor

    @exergy_factor.setter
    def exergy_factor(self, exergy_factor):
        DataAccessors.type_checker(exergy_factor, self, "exergy_factor", "numeric")
        self._exergy_factor = exergy_factor






class FuelVector(Element):
    # Sources:
    # - natural gas: https://www.techniques-ingenieur.fr/res/pdf/encyclopedia/tiabl-bm2591-version1.pdf /
    # "Gaz naturel - Energie fossile" / Richard TILAGONE
    # - biomass (wood): https://www.techniques-ingenieur.fr/res/pdf/encyclopedia/42593210-be8750.pdf /
    # "Production de chaleur à partir de bois - Émissions atmosphériques. Notions de base" / Erwan AUTRET, Yann RogAUME
    #     Other possibility from:
    #         - specific exergy factor (kJ/kg): Classic Biomass > Wood (Lizarraga et Picallo-Perez, 2020, page 242)
    #         hypothesis: given for completely dry wood
    #         - LHV (kWh/ton): 'Moyenne Feuillus durs, PCI anhydre'
    #         https://www.ademe.fr/sites/default/files/assets/documents/referentiels-combustibles-bois-energie-
    #         plaquettes-forestieres-200804.pdf page 42
    #         Similar to http://mbaudin.free.fr/bois/combustible_bois.pdf page 12
    typical_flame_temperature = {"Natural gas": (1690 + 273),  # Carnot factor = 84.8% at 25°C
                                 "Biomass": (800 + 273)}       # Carnot factor = 72.2% at 25°C

    def __init__(self, exergy_factor, name=None):
        """
        A FuelVector instance may describe every energy vector whose exergy factor does not depend on a temperature
        (i.e. ThermalVector, ThermalVectorPair) or is not unity (i.e. ElectricityVector).
        Power is exchanged in kW.

        These vectors are usually associated with a combustion flame temperature which can be used by the user
        to define the exergy factor according to:
        exergy_factor = 1 - dead_state_temperature / flame temperature.

        Parameters
        ----------
        exergy_factor : float
            0 <= exergy_factor
        name : str, optional

        Notes
        -----
        1. The dead state temperature is the reference temperature for calculation of exergy factor of ThermalVectorPair and ThermalVector instances.
        It can be modified using the `set_dead_state_temperature` function of tamos.element.
        2. Some typical flame temperatures are stored in the class attribute `FuelVector.typical_flame_temperature` (natural gas, biomass).
        3. Power flows related to a FuelVector take for reference the lower heating value of the fuel. This impacts the efficiency
        models in production components.
        4. The exergy factor can be redefined using the `exergy_factor` attribute.

        """
        if name is None:
            name = "FuelVector"
        super().__init__(name)
        self.exergy_factor = exergy_factor


class ElectricityVector(Element):
    def __init__(self, name=None):
        """
        ElectricityVector instances are only defined by an exergy factor (default to 1).
        Power is exchanged in kW.

        Parameters
        ----------
        name : str, optional

        Notes
        -----
        The exergy factor can be redefined using the `exergy_factor` attribute.

        """
        if name is None:
            name = "Electricity"
        super().__init__(name)
        self.exergy_factor = 1


class ThermalElement(Element):
    pass

class ThermalVector(ThermalElement):
    def __init__(self, temperature, name=None):
        """
        ThermalVector instances may:

        * be used in a ThermalVectorPair instance.
        * describe an infinite thermal reservoir.
          It is used to model a thermal source or sink that would not be affected by the energy taken from or given to it.

          >>> ThermalVector(temperature=temperature_profile)

          For example, it is relevant to model the ambiant air using a ThermalVector. Such a vector, associated with
          a Grid instance (i.e. an element_IO component), would describe the heat rejected by heat pumps on their condenser side.

          >>> Grid(element=air, exergy_count=False)

          In that case, exergy count migh not be relevant thus it can be disabled using the `exergy_count` argument of the Grid
          class.

        Power is exchanged in kW.

        Parameters
        ----------
        temperature : int, float or numpy.ndarray
            In Kelvins (K).
            Used in:

            * the efficiency definition of some production components (e.g. heat pumps)
            * the `exergy_factor` attribute
            * power and mass balance if this instance is used in a THermalVectorPair

        name : str, optional

        Notes
        -----
        1. Two ThermalVector instances having the same temperature are still considered different.
        2. The dead state temperature is the reference temperature for calculation of exergy factor of ThermalVectorPair and ThermalVector instances.
           It can be modified using the `set_dead_state_temperature` function of tamos.element.
           The exergy factor calculation is done as follow:
           exergy_factor = 1 - dead_state_temperature / temperature
        3. The exergy factor can be user-defined using the `exergy_factor` attribute.
        """
        if name is None:
            name = "TV"
        super().__init__(name)
        self.temperature = temperature
        self._used_in_TVP = 0

    @property
    def temperature(self):
        """
        int, float or numpy.ndarray
        In Kelvins (K)

        """
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        DataAccessors.type_checker(temperature, self, "temperature", "numeric")
        old_temperature = None
        if hasattr(self, "_temperature"):
            old_temperature = self.temperature

        self._temperature = temperature
        self.exergy_factor = 1 - Element._get_dead_state_temperature() / self.temperature
        for TVP in ThermalVectorPair._existing_TVPs.values():
            if self in TVP.get_vectors():
                is_cooled_before = TVP.is_cooled
                TVP._set_temperature_attr()
                is_cooled_after = TVP.is_cooled
                if is_cooled_after != is_cooled_before:
                    self._temperature = old_temperature
                    TVP._set_temperature_attr()
                    status = "cooled down" if is_cooled_before else "warmed up"
                    raise ValueError(f"{self}: Cannot update temperature as ThermalVectorPair {TVP}"
                                     f" would not be {status} anymore.")
                else:
                    print(f"{self}: ThermalVectorPair {TVP} was updated.")

class ThermalVectorPair(ThermalElement):
    _existing_TVPs = {}

    #  todo: replace __init__ by fetch using __new__
    @staticmethod
    def _fetch(in_TV, out_TV, Cp=None, name=None):
        """
        This method defines a ThermalVectorPair instance by its incoming and outcoming ThermalVector instances.

        A ThermalVectorPair instance aims to represent a fluid (typically water) receiving or giving thermal energy
        to a subsystem by going through a sensible heat exchange. The fluid entering the subsystem is described by ThermalVector
        `in_TV` with temperature in_TV.temperature. The fluid exiting the subsystem is `out_TV`, with temperature out_TV.temperature.
        Power is exchanged in kW.


        Parameters
        ----------
        in_TV : ThermalVector

        out_TV : ThermalVector

        Cp : int, float or numpy.ndarray, optional
        
        name : str, optional

        Returns
        -------
        * If a ThermalVectorPair instance defined by (in_TV, out_TV) already exists, returns this instance

          * If `name` is None, current name of the instance is kept.
          * Else, the instance is renamed.

        * Else, returns a new ThermalVectorPair instance.

        Notes
        -----
        1. It is convenient to speak about ThermalVectorPair as 'TVP'.
        2. A positive power flow associated with a TVP (in_TV, out_TV) in a component describes
           a positive mass flow rate of ThermalVector in_TV (mass enters the system)
           and the negated same mass flow rate of ThermalVector out_TV (mass leaves the system).
        3. If (in_TV, out_TV) defines a ThermalVectorPair instance `TVP1`, the ThermalVectorPair (out_TV, in_TV) is called
           'the invert of `TVP1`', for example `TVP2`, and can be accessed following two ways:

           >>> TVP2_first = fetch_TVP(in_TV=TVP1.out_TV, out_TV=TVP1.in_TV)
           >>> TVP2_second = ~TVP1
           >>> TVP2_first is TVP2_second
           True

        4. Though balances are based on power flows in kW at the component scale, `Cp` is required to perform mass balances at hub interface scale.
        5. Cp should depend on the temperature of `in_TV` and `out_TV`. In practice, Cp variations are small on the temperature ranges [0°C, 110°C].
           e.g.

           * max: Cp(T=110°C)=4228.3 J/(kg.K)
           * min: Cp(T=40°C)=4179.6 J/(kg.K)

        6. The `exergy_factor` attribute is calculated as follow:
           exergy_factor = 1 - dead_state_temperature * log(out_TV.temperature/in_TV.temperature) / (out_TV.temperature-in_TV.temperature)

        """
        if (in_TV, out_TV) in ThermalVectorPair._existing_TVPs:
            TVP = ThermalVectorPair._existing_TVPs[(in_TV, out_TV)]
            if name is not None:
                TVP.name = name
        else:
            invert_exists = (out_TV, in_TV) in ThermalVectorPair._existing_TVPs
            if invert_exists:
                invert_TVP = ThermalVectorPair._existing_TVPs[(out_TV, in_TV)]
                if name is None:
                    name = "!" + invert_TVP.name
                if Cp is not None:
                    if Cp != invert_TVP.Cp:
                        print(f"Warning: {ThermalVectorPair}: TVP with new name '{name}' already exists thus "
                              f"new value for 'Cp' is ignored. ")
                Cp = invert_TVP.Cp
                TVP = ThermalVectorPair(in_TV, out_TV, Cp, name)
                TVP._invert_matcher = invert_TVP
                                                            # ~TVP does exist thus TVP is identified as its invert
                                                            # for postprocessing in ResultsExport
            else:
                if name is None:
                    name = f"{in_TV.name}_{out_TV.name}"
                if Cp is None:
                    Cp = 4200 / 3.6e6
                TVP = ThermalVectorPair(in_TV, out_TV, Cp, name)
                TVP._invert_matcher = TVP                   # ~TVP does not exist thus TVP is the reference among (TVP, ~TVP),
                                                            # with an invert being itself for ease of implementation in ResultsExport
            ThermalVectorPair._existing_TVPs[(in_TV, out_TV)] = TVP
        return TVP

    @staticmethod
    def _get_existing_TVPs():
        return ThermalVectorPair._existing_TVPs

    def __init__(self, in_TV, out_TV, Cp, name):
        """
        Do not use. See `fetch_TVP` function.

        """
        for arg, attr_name in zip([in_TV, out_TV], ["in_TV", "out_TV"]):            # DataAccessors.type_checker unused to prevent ThermalVector check (._used_in_TVP)
            if not isinstance(arg, ThermalVector):
                raise TypeError(f"ThermalVectorPair: Invalid data type for '{attr_name}'.")
        super().__init__(name)
        assert in_TV != out_TV, f"{self}: 'in_TV' and 'out_TV' must be different."
        self._in_TV = in_TV
        self._out_TV = out_TV
        self.Cp = Cp
        self._in_TV._used_in_TVP += 1
        self._out_TV._used_in_TVP += 1

    @property
    def in_TV(self):
        """
        Entering ThermalVector instance.

        """
        return self._in_TV

    @property
    def out_TV(self):
        """
        Exiting ThermalVector instance.

        """
        return self._out_TV

    @property
    def is_cooled(self):
        """
        True if the temperature of `in_TV` is always greater than the one of `out_TV`.
        False otherwise.

        """
        return self._is_cooled

    @property
    def is_warmed(self):
        """
        True if the temperature of `in_TV` is always smaller than the one of `out_TV`.
        False otherwise

        """
        return self._is_warmed

    @property
    def Cp(self):
        """
        Specific heat capacity, in kWh/(kg.K).
        int, float or numpy.ndarray

        """
        return self._Cp

    @Cp.setter
    def Cp(self, Cp):
        DataAccessors.type_checker(Cp, self, "Cp", "numeric")
        self._Cp = Cp
        self._set_temperature_attr()

    @property
    def DT(self):
        """
        Temperature difference between the two thermal vectors.

        DT = in_TV.temperature - out_TV.temperature
        if `is_cooled`, DT >= 0.
        Else, DT <= 0

        """
        return self._DT

    def _set_temperature_attr(self):
        # Do not change. Deals with case DT = 0 by setting both "is_cooled" and "is_warmed" to True
        in_TV, out_TV = self._in_TV, self._out_TV
        self._DT = in_TV.temperature - out_TV.temperature
        self._is_cooled = False
        self._is_warmed = False
        positive = all(self._DT >= 0)
        negative = all(self._DT <= 0)
        possible_zero = any(self._DT == 0)
        if positive:
            self._is_cooled = True
            
        if negative:
            self._is_warmed = True
            self._DT = - self._DT

        if not (positive or negative):
            raise AttributeError(f"{self}: Difference of temperature between "
                                   f"incoming and outcoming thermal vectors must be either "
                                       f"always positive or always negative.")
        if possible_zero:
            print(f"Warning: {self}: Difference of temperature between incoming and outcoming thermal vectors "
                             f"is sometimes 0, which can be conflictual in some components models.")

        self._CpDT = self.Cp * self._DT
        self._DT_LM = (out_TV.temperature - in_TV.temperature) / log((out_TV.temperature / in_TV.temperature))
        self.exergy_factor = 1 - Element._get_dead_state_temperature() / self._DT_LM

    def get_vectors(self):
        """

        Returns
        -------
        A list of the ThermalVector instances associated with this instance, [in_TV, out_TV].

        """
        return [self._in_TV, self._out_TV]

    def get_cold_vector(self):
        """
        Returns
        -------
        `in_TV` if its temperature is smaller than the one of `out_TV`, else `out_TV`

        """
        return self._in_TV if self.is_warmed else self._out_TV

    def get_hot_vector(self):
        """
        Returns
        -------
        `in_TV` if its temperature is greater than the one of `out_TV`, else `out_TV`

        """
        return self._in_TV if self.is_cooled else self._out_TV

    def _compute_flow_rate(self, power, t):
        """
        Helper method, for advanced users
        """
        return DataAccessors.get2(power / self._CpDT, t, t + 1)

    def _compute_power(self, flow_rate, t):
        """
        Helper method, for advanced users
        """
        return DataAccessors.get2(flow_rate * self._CpDT, t, t + 1)

    def __invert__(self):
        return ThermalVectorPair._fetch(self._out_TV, self._in_TV)
