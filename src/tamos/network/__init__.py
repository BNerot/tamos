from .HRE_thermal_network import HREThermalNetwork
from .non_thermal_network import NonThermalNetwork
from .thermal_network import ThermalNetwork
from .network_base import Network as _Network
set_distance_function = _Network._set_distance_function
get_distance_function = _Network._get_distance_function


