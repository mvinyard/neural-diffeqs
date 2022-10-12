
__module_name__ = "_base_NDE.py"
__doc__ = """Base classes for all models."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base_ancilliary._base_support import parse_passed_networks


# -- main module class: ------------------------------------------------------------------
class BaseNDE(ABC, torch.nn.Module):
    """
    Abstract base class for NeuralDiffEq. Most shared common ancestor class.
    Common to all NDE functions.
    """

    def __init__(self):
        super(BaseNDE, self).__init__()
        self.__dict__.update(locals())

    def __assemble__(self):
        parse_passed_networks(self, self.networks)
        self.__setup__()
    
    @abstractmethod
    def __setup__(self):
        pass

    def potential(self, net, x):
        """
        Pass through potential net
        Note:
        -----
        (1) I'd need to look deeper, but if memory serves, there is a good
            reason for dedicating an entire line to x = x.requires_grad_()
            rather than simply returning net(x.requires_grad_()).
        """
        x = x.requires_grad_()
        return net(x)

    def potential_pass(self, net, x):
        """
        Return the drift position as the gradient of the potential.
        """
        potential = self._potential(net, x)
        return torch.autograd.grad(
            potential, x, torch.ones_like(potential), create_graph=True
        )[0]

    def forward_pass(self, net, x):
        """
        If not a potential_net, simply pass the torch.nn.Module
        through the network.
        """
        return net(x)
