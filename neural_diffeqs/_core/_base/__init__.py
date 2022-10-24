
__module_name__ = "__init__.py"
__doc__ = """__init__.py module"""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import base classes: ----------------------------------------------------------------
from ._base_NDE import BaseNDE
from ._base_SDE import BaseSDE
from ._base_ODE import BaseODE


# -- import supporting functions: --------------------------------------------------------
from ._base_ancilliary._instantiate_mu_sigma_networks import instantiate_mu_sigma_networks
from ._base_ancilliary._base_support import (
    as_list,
    set_state_sizes,
    is_potential_net,
    define_forward_function,
    parse_passed_networks,
)