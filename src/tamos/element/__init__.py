from .element import ElectricityVector, FuelVector, ThermalElement, ThermalVector, ThermalVectorPair as _ThermalVectorPair, Element as _Element
get_dead_state_temperature = ThermalElement._get_dead_state_temperature
set_dead_state_temperature = ThermalElement._set_dead_state_temperature
fetch_TVP = _ThermalVectorPair._fetch
get_existing_TVPs = _ThermalVectorPair._get_existing_TVPs
del ThermalElement


