

# -- import packages: ----------------------------------------------------------
import torch
import ABCParse
import torch_nets


# -- import standard libraries: ------------------------------------------------
from abc import abstractmethod


# -- import local dependencies: ------------------------------------------------



# -- Main operational class: ---------------------------------------------------
class BaseNeuralSDE(torch.nn.Module, ABCParse.ABCParse):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        """
        Must call self.__config__(locals()) in the __init__ of theinheriting
        class. 
        """

    def __config__(self, kwargs):
        self.__parse__(kwargs=kwargs)

        """Sets up mu and sigma given params"""
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
            out_features=self.state_size * self.brownian_dim,
            hidden=self.sigma_hidden,
            activation=self.sigma_activation,
            dropout=self.sigma_dropout,
            bias=self.sigma_bias,
            output_bias=self.sigma_output_bias,
        )

    # -- required methods in child classes: ------------------------------------
    @abstractmethod
    def drift(self):
        """Called by self.f"""
        ...

    @abstractmethod
    def diffusion(self):
        """Called by self.g"""
        ...

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Should return the output of self.drift"""
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Should return the output of self.diffusion"""
        return self.diffusion(y)
