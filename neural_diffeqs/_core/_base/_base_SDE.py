
__module_name__ = "_base_SDE.py"
__doc__ = """Base class for Neural SDE models."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base_NDE import BaseNDE
from ._base_ancilliary._base_support import define_forward_function, parse_passed_networks


# -- BaseSDE: ----------------------------------------------------------------------------
class BaseSDE(BaseNDE):
    """Abstract class for BaseSDE. torchsde.sdeint requires f and g methods"""
    def __init__(self):
        super(BaseSDE, self).__init__()

    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def g(self):
        pass

    def __setup__(self):
        parse_passed_networks(self, self.networks)
        define_forward_function(self, name="mu")
        define_forward_function(self, name="sigma")

    def view_g_state(self, y):
        return y.view(y.shape[0], y.shape[1], self.brownian_size)
