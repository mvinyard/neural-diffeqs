
__module_name__ = "_base_ODE.py"
__doc__ = """Base class for Neural ODE models."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base_NDE import BaseNDE
from ._base_ancilliary._base_support import define_forward_function


# -- BaseODE: ----------------------------------------------------------------------------
class BaseODE(BaseNDE):
    """Abstract class for BaseODE"""
    """Abstract class for BaseODE. torchdiffeq.odeint requires `forward` method"""


    def __init__(self):
        super(BaseODE, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    def __setup__(self):
        define_forward_function(self, name="mu")
    