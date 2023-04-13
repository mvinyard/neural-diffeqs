
# -- import packages: ----
import torch
import ABCParse
import torch_nets

from ._potential import Potential


# -- define types: -------
from typing import Union, List, Any
NoneType = type(None)


# -- SDE class: -----
class PotentialNeuralSDE(torch.nn.Module, ABCParse.ABCParse):
    def __init__(
        self,
        state_size,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        sigma_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        sigma_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        sde_type="ito",
        noise_type="general",
        brownian_dim=1,
    ):
        super().__init__()

        self.__parse__(kwargs=locals())

        self.mu = torch_nets.TorchNet(
            in_features=self.state_size,
            out_features=self.state_size,
            hidden=self.mu_hidden,
            activation=self.mu_activation,
            dropout=self.mu_dropout,
            bias=self.mu_bias,
            output_bias=self.mu_output_bias,
        )
        self.sigma = torch_nets.TorchNet(
            in_features=self.state_size,
            out_features=self.state_size * brownian_dim,
            hidden=self.sigma_hidden,
            activation=self.sigma_activation,
            dropout=self.sigma_dropout,
            bias=self.sigma_bias,
            output_bias=self.sigma_output_bias,
        )

        self.potential = Potential(self.mu.out_features)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Should return the output of self.drift"""
        return self.mu(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Should return the output of self.diffusion"""
        return self.sigma(y).view(y.shape[0], y.shape[1], self.brownian_dim)

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Should return the output of self.prior_drift"""
        return self.potential.gradient(self.mu(y))
