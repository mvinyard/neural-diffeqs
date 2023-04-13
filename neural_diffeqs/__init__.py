
__module_name__ = "__init__.py"
__doc__ = """__init__.py module"""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])
__version__ = "0.2.2rc"



# from ._neural_ode import NeuralODE

# -- import subpackages: -------------------------------------------------------
from . import core, models


# -- import modules: -----------------------------------------------------------
from .models import NeuralSDE, AugmentedNeuralSDE
from .core import base_models

# from ._potential import Potential
# from ._potential_neural_sde import PotentialNeuralSDE