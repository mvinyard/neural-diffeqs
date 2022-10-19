
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])
__version__ = "0.2.0"


# -- import modules: ---------------------------------------------------------------------
from ._core import _base as base
from ._core._neural_SDE import NeuralSDE
from ._core._neural_ODE import NeuralODE


# -- import packages: --------------------------------------------------------------------
from torch_composer import TorchNet