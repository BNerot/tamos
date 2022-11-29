from .data_IO.data_IO import DataAccessors, NamesFormatter


class MetaComponent:
    _allow_duplicated_names_ = False
    _names = {""}
    _names_counter = {}

    @staticmethod
    def _allow_duplicated_names(allow):
        """
        Define the policy regarding components sharing the same name.

        Parameters
        ----------
        allow: bool

            * If True, duplicated names might exist.
              Leads to problems in model export regarding the identification of different MILP objects.
            * If False, names are changed during the component declaration so that no duplicate exists.

        """
        DataAccessors.type_checker(allow, MetaComponent, "allow", bool)
        if allow:
            print(f"Warning: already used names may be used once again."
                  f"This might lead to problems regarding KPI declaration and results export.")
        else:
            print(f"Warning: already used names are made different using a dedicated integer counter.")
        MetaComponent._allow_duplicated_names_ = allow

    @staticmethod
    def _reset():
        """
        Forget every component name.

        """
        MetaComponent._names = {""}
        MetaComponent._names_counter = {}

    def __init__(self, name):
        self._name = None
        self.name = name

    @property
    def name(self):
        """
        Name of the instance

        This name is used in MILP model description.

        str
        names must not exceed 255 characters,
        all of which must be alphanumeric (a-z, A-Z, 0-9) or one of these symbols:
        ! " # $ % & , . ; ? @ _ ‘ ’ { } ~.

        """
        return self._name

    @name.setter
    def name(self, name):
        name = str(name)
        current_name = self.name
        if current_name != name:
            if name in MetaComponent._names:
                if not MetaComponent._allow_duplicated_names_:
                    if name in MetaComponent._names_counter:
                        MetaComponent._names_counter[name] += 1
                    else:
                        MetaComponent._names_counter[name] = 2
                    idx = MetaComponent._names_counter[name]
                    new_name = f"{name}{idx}"
                    self.name = new_name
                    return None

            if any([separator in name for separator in [NamesFormatter._separator_right,
                                                        NamesFormatter._separator_left,
                                                        NamesFormatter._separator_both]]):
                raise ValueError(f"Component of type {self.__class__.__name__} cannot be named '{name}':"
                                 f"caracters '{NamesFormatter._separator_right}' and "
                                 f"'{NamesFormatter._separator_left}' are forbidden")

            self._name = name
            MetaComponent._names.add(name)
            if current_name is not None:
                MetaComponent._names.remove(current_name)

    def _change_set_attribute(self, attr_name, objects, add, remove):
        current_objects = getattr(self, attr_name)
        if objects is None:
            objects = set()
        else:
            objects = set(objects)
        if add:
            setattr(self, attr_name, current_objects.union(objects))
        elif remove:
            setattr(self, attr_name, current_objects.difference(objects))
        else:
            setattr(self, attr_name, objects)

    def __repr__(self):
        return self._name


class Component(MetaComponent):

    def __init__(self, name, properties):
        DataAccessors.type_checker(properties, name, "properties", dict)
        super().__init__(name)
        self.properties = {"Discount rate (%)": 3.5, "Lifetime": 40}
        if properties is not None:
            self.properties.update(properties)
        self._used_elements = set()

    @staticmethod
    def _actu_component(period, discount_rate):
        return (1 + discount_rate) ** period

    def compute_actualized_cost(self,
                                CAPEX,
                                OPEX,
                                system_lifetime,
                                lifetime = None,
                                discount_rate = None):
        """
        Compute the cost of a component using its 'Lifetime' and 'Discount rate (%)' properties.

        Parameters
        ----------
        CAPEX : float
            Capital Expenditure. Cost in euros paid every `technical_lifetime` periods.
        OPEX : float
            Operational Expenditure. Cost in euros paid each period.
        system_lifetime : int
            Number of periods defining the existence of the energy system.
        lifetime: int, optional
            Number of periods defining the existence of the component.
            If specified, overwrite the "Lifetime" property.
        discount_rate: float
            In percent (%). Describes the importance of the economic amortization process, per period.
            If specified, overwrite the "Discount rate (%)" property.

        Returns
        -------
        A 3-tuple (total_cost, CAPEX_share, OPEX_share) where:

        * CAPEX_share is the share of total cost related to `CAPEX`
        * OPEX_share is the share of total cost related to `OPEX`
        * total_cost = CAPEX_share + OPEX_share


        Notes
        -----
        Takes into account residual value of component in the case `system_lifetime` is not a multiple of `lifetime`.
        In this case, the last replacement occuring at period replacement_period is paid in proportion of 'CAPEX'
        depending linearly on the number of periods left:
        CAPEX * (system_lifetime - replacement_period) / lifetime

        """
        if lifetime is None:
            lifetime = self._get_property("Lifetime")
        if discount_rate is None:
            discount_rate = self._get_property("Discount rate (%)")

        discount_rate /= 100
        CAPEX__ = CAPEX  # initial investment
        OPEX__ = 0
        for period in range(1, system_lifetime + 1):  # at the end of period "period"
            if period % lifetime == 0:
                if system_lifetime - period >= lifetime:  # not the last replacement
                    CAPEX_ = CAPEX
                else:  # last replacement: residual value once system_lifetime has been reached
                    CAPEX_ = CAPEX * (system_lifetime - period) / lifetime
            else:
                CAPEX_ = 0
            actu = Component._actu_component(period, discount_rate)
            CAPEX__ += CAPEX_ / actu
            OPEX__ += OPEX / actu
        # Basic test:
        # try with discount_rate=0 with a CAPEX of n € and a technical lifetime of n periods:
        # the system cost 1€ per period
        # so total_cost = system_lifetime
        total_cost = CAPEX__ + OPEX__
        return total_cost, CAPEX__, OPEX__

    def _get_property(self, property_):
        return self.properties[property_]

    @property
    def used_elements(self):
        """
        Elements used by the component.

        """
        return self._used_elements

    def _add_used_element(self, element, attr_name, kind):
        DataAccessors.type_checker(element, self, attr_name, kind)
        if element in self._used_elements:
            raise AttributeError(f"{self}: Element '{element}' cannnot be used twice in the same component.")
        try:
            bool_ = element.__invert__() in self._used_elements
        except AttributeError as e:
            bool_ = False
        if bool_:
            print(self._used_elements)
            raise AttributeError(f"{self}: "
                                 f"ThermalVectorPair '{element}' and '{element.__invert__()}'"
                                 f" cannnot be used in the same component.")

        setattr(self, f"_{attr_name}", element)
        self._used_elements.add(element)
