"""
tamos is a modelling software dedicated to the study of thermal architectures.

Available subpackages
---------------------
data_IO
    Export and import optimization results.
element
    Define energy vectors.
elementIO
    Components that connect the energy system with its environment.
network
    Components that connect energy hubs with each other.
production
    Components that convert elements into other ones.
storage
    Components that store elements.
solve_tools
    Convenient helper functions to run optimizations.


Available classes
-----------------
Hub
    Create energy hubs.
InterfaceMask
    Set constraints on power and energy flows received by a component.
TimeSettings
    Configure the operation and sizing temporal scales for optimization.
MILPModel
    Gather energy hubs and settings in an energy system for optimization.

Utilities
---------
allow_duplicated_names
    Change the policy regarding components having the same name.
reset_names_list
    Forget every already defined name.
use_name_in_MILP
    Make complete variables and constraints names available for debugging a LP or MPS file.
"""

from .data_IO.data_IO import NamesFormatter
import tamos.data_IO
import tamos.element
import tamos.elementIO
from tamos.hub import Hub
from tamos.mask import InterfaceMask
import tamos.network
from tamos.optimization import TimeSettings, MILPModel
import tamos.production
import tamos.storage
import tamos.solve_tools
from tamos.component import MetaComponent

allow_duplicated_names = MetaComponent._allow_duplicated_names
reset_names_list = MetaComponent._reset
use_name_in_MILP = NamesFormatter.use_name_in_MILP
__version__ = "0.1.0"


del MetaComponent, NamesFormatter
