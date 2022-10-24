
__module_name__ = "__init__.py"
__doc__ = """__init__.py module"""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import modules: ---------------------------------------------------------------------
from ._core import _base as base
from ._core._neural_SDE import NeuralSDE
from ._core._neural_ODE import NeuralODE


# -- import packages: --------------------------------------------------------------------
from torch_composer import TorchNet