
__module_name__ = "_NeuralSDE.py"
__doc__ = """ To-Do: write this."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
import torch


# -- import packages: --------------------------------------------------------------------
from ._base import BaseSDE, instantiate_mu_sigma_networks


# -- import packages: --------------------------------------------------------------------
class NeuralSDE(BaseSDE):
    def __init__(
        self,
        state_size: 50,
        hidden={1: [200, 200]},
        activation_function=torch.nn.Tanh,
        dropout=0,
        input_bias=True,
        output_bias=True,
        brownian_size: int = 1,
        mu_init: float = None,
        sigma_init: float = None,
        noise_type: str = "general",
        sde_type: str = "ito",
        **kwargs,
    ):
        self.__dict__.update(locals())
        super(NeuralSDE, self).__init__()

        self.networks = instantiate_mu_sigma_networks(
            state_size=state_size,
            hidden=hidden,
            activation_function=activation_function,
            dropout=dropout,
            input_bias=input_bias,
            output_bias=output_bias,
            **kwargs,
        )
        self.__setup__()

    # -- drift: -------------------------------------------------
    def f(self, t, y0):
        return self.mu_forward(self.mu, y0)

    # -- diffusion: ---------------------------------------------
    def g(self, t, y0):
        return self.view_g_state(self.sigma_forward(self.sigma, y0))
