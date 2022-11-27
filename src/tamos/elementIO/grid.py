from .elementIO_base import ElementIO


class Grid(ElementIO):
    def __init__(self,
                 element,
                 emissions=None,
                 element_cost=None,
                 exergy_count=True,
                 name=None):
        """
        Allows unconstrained element exchanges between the energy system and its environment.

        Grid components are associated with the following exported decision variables:

        * X_EXT, binary.
          Whether the Grid instance is used by the hub.
        * For all t, F_EXT(t), continuous, in kW.
          The power related to `element` entering the grid (i.e. leaving the hub interface).

        Grid components declare the following KPIs:

        * `ElementIO CO2`
          In kgEqCO2.
          Defines the "CO2" objective function. Related to the `emissions` attribute.
        * `ElementIO Exergy`
          In kWh.
          Defines the "Exergy" objective function. Related to the `exergy_factor` attribute of `element`.

        Parameters
        ----------
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

        """
        if name is None:
            name = f"Grid..{element}"
        super().__init__(element=element,
                         emissions=emissions,
                         element_cost=element_cost,
                         exergy_count=exergy_count,
                         name=name)
