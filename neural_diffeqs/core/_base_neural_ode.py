
import torch
import ABCParse
from abc import abstractmethod

from ._base_neural_diffeq import BaseDiffEq
from ._diffeq_config import DiffEqConfig

class BaseODE(BaseDiffEq):
    DIFFEQ_TYPE = "ODE"

    def __init__(self, *args, **kwargs):
        super().__init__()

        """
        Must call self.__config__(locals()) in the __init__ of theinheriting
        class.
        
        
        """

    def __config__(self, kwargs):
        """Sets up mu and sigma given params"""

        self.__parse__(kwargs=kwargs)

        self._config_kwargs = ABCParse.function_kwargs(func=DiffEqConfig, kwargs=kwargs)
        configs = DiffEqConfig(**self._config_kwargs)
        self.mu = configs.mu
    @property
    def device(self):
        return list(self.parameters())[0].device

    # -- required methods in child classes: ------------------------------------
    @abstractmethod
    def drift(self):
        """Called by self.f and/or self.forward"""
        ...
        
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.drift(y)

    def diffusion(self, y):
        # keep for compatibility with torchsde.sdeint
        """Called by self.g"""
        return torch.zeros([y.shape[0], y.shape[1], self.brownian_dim], device=self.device)
        
    
