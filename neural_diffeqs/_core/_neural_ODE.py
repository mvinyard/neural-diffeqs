
__module_name__ = "_NeuralODE.py"
__doc__ = """Neural SODE module. Contains API-facing NeuralODE class."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import packages: --------------------------------------------------------------------
import torch
import torch_composer


# -- import local dependencies: ----------------------------------------------------------
from ._base import BaseODE


# -- API-facing class: -------------------------------------------------------------------
class NeuralODE(BaseODE):
    """forward is a required method for torchdiffeq.odeint"""
    def __init__(
        self,
        state_size: int,
        hidden: dict({int: [int, int], int: [int, int]}) = {1: [200, 200]},
        activation_function: 'torch.nn.modules.activation.<func>' = torch.nn.Tanh,
        potential_net: bool = False,
        dropout: float = 0,
        input_bias: bool = True,
        output_bias: bool = True,
    ):
        """
        Instantiate a NeuralODE.

        Parameters:
        -----------
        state_size
            type: int

        hidden
            type: dict
            default: {1: [200, 200]}

        activation_function:
            type: 'torch.nn.modules.activation.<func>'
            default: torch.nn.Tanh,

        potential_net
            If True, overrides out_dim and output_bias, setting them to 1 and False, respectively.
            type: bool
            default: False

        dropout
            type: float
            default: 0

        input_bias
            type: bool
            default: True

        output_bias
            type: bool
            default: True

        Returns:
        --------
        None
        
        Notes:
        ------

        """
        super(NeuralODE, self).__init__()

        self.mu = torch_composer.TorchNet(
            in_dim=state_size,
            out_dim=state_size,
            hidden=hidden,
            activation_function=activation_function(),
            potential_net=potential_net,
            dropout=dropout,
            input_bias=input_bias,
            output_bias=output_bias,
        )

        self.__setup__()

    def forward(self, t, y0):
        """
        Forward (drift, deterministic) method, core to NeuralODE.

        Parameters:
        -----------
        t
            time tensor of shape: (t,)
            type: torch.Tensor
            default: None

        y0
            state vector
            type: torch.Tensor

        Returns:
        --------
        forward(y0)
            drift: Tensor of same shape as y0.
            type: torch.Tensor

        Notes:
        ------
        (1) t required syntactically though not functionally.

        Examples:
        ---------
        >>> func = NeuralODE(state_size)
        >>> x_hat_f = func.forward(None, x)
        """
        return self.mu_forward(self.mu, y0)