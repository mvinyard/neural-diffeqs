
# -- import packages: ----------------------------------------------------------
import torch
import ABCParse


# -- import local dependencies: ------------------------------------------------
from .core._base_neural_sde import BaseSDE


# -- import standard libraries and define types: -------------------------------
from typing import Union, List, Any
NoneType = type(None)


# -- Main operational class: ---------------------------------------------------
class NeuralSDE(BaseSDE):
    DIFFEQ_TYPE = "SDE"
    def __init__(
        self,
        state_size: int,
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
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type: str = "ito",
        noise_type: str = "general",
        brownian_dim: int = 1,
        coef_drift: float = 1.,
        coef_diffusion: float = 1.,
    ) -> torch.nn.Module:
        """
        Parameters
        ----------
        state_size: int
            Input and output state size of the differential equation.
        
        mu_hidden: Union[List[int], int], default = [2000, 2000]
            Architecture of the hidden layers of the drift neural network.
        
        sigma_hidden: Union[List[int], int], default = [400, 400]
            Architecture of the hidden layers of the diffusion neural network.
            
        mu_activation: Union[str, List[str]], default = "LeakyReLU"
        
        sigma_activation: Union[str, List[str]], default = "LeakyReLU"
        mu_dropout: Union[float, List[float]], default = 0.2
        sigma_dropout: Union[float, List[float]], default = 0.2
        mu_bias: bool, default = True
        sigma_bias: List[bool], default = True
        mu_output_bias: bool, default = True
        sigma_output_bias: bool, default = True
        mu_n_augment: int, default = 0
        sigma_n_augment: int, default = 0
        sde_type: str, default = "ito"
        noise_type: str, default = "general"
        brownian_dim: int, default = 1
        coef_drift: float, default = 1.
        coef_diffusion: float, default = 1.
        
        Returns
        -------
        NeuralSDE: torch.nn.Module
        
        Notes
        -----
        """
        super().__init__()

        self.__config__(locals())

    def drift(self, y)->torch.Tensor:
        return self.mu(y) * self._coef_drift

    def diffusion(self, y)->torch.Tensor:
        return self.sigma(y).view(y.shape[0], y.shape[1], self._brownian_dim) * self._coef_diffusion
