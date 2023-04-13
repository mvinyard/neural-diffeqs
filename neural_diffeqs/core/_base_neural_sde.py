

# -- import packages: ----------------------------------------------------------
import torch
import ABCParse


# -- import standard libraries: ------------------------------------------------
from abc import abstractmethod


# -- import local dependencies: ------------------------------------------------
from ._diffeq_config import DiffEqConfig


# -- Main operational class: ---------------------------------------------------
class BaseNeuralSDE(torch.nn.Module, ABCParse.ABCParse):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        """
        Must call self.__config__(locals()) in the __init__ of theinheriting
        class.
        
        
        """

    def __config__(self, kwargs):
        """Sets up mu and sigma given params"""
        
        self.__parse__(kwargs=kwargs)
        
        self._config_kwargs = ABCParse.function_kwargs(func = DiffEqConfig, kwargs=kwargs)
        self.mu, self.sigma = DiffEqConfig(**self._config_kwargs)()


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
