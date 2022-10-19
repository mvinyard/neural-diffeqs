
# -- import packages: -----------------------
import torch
import torch_composer


# -- import local dependencies: -----------------------
from ._base import BaseODE


class NeuralODE(BaseODE):
    def __init__(
        self,
        state_size,
        hidden={1: [200, 200]},
        activation_function=torch.nn.LeakyReLU(),
        dropout=0,
        input_bias=True,
        output_bias=True,
    ):
        super(NeuralODE, self).__init__()

        self.mu = torch_composer.TorchNet(
            in_dim=state_size,
            out_dim=state_size,
            hidden=hidden,
            activation_function=activation_function,
            dropout=dropout,
            input_bias=input_bias,
            output_bias=output_bias,
        )

        self.__setup__()

    def forward(self, t, y0):
        return self.mu_forward(self.mu, y0)