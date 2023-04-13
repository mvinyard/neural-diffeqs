
# -- import packages: ----------------------------------------------------------
import torch


# -- operational class: --------------------------------------------------------
class Potential(torch.nn.Sequential):

    """Linear transform state of arbitrary dimension to a 1-D potential value"""

    def __init__(self, state_size: int):
        """
        Parameters:
        -----------
        state_size
            type: int

        Returns:
        --------
        None, instantiates class.
        """
        super().__init__()
        self.add_module("psi", torch.nn.Linear(state_size, 1, bias=False))

    def _gradient(self, ψ, y):
        return torch.autograd.grad(ψ, y, torch.ones_like(ψ), create_graph=True)[0]

    def gradient(self, y):
        """in this use-case, y is likely the output of a neural network"""
        y = y.requires_grad_()
        return self._gradient(self.psi(y), y)