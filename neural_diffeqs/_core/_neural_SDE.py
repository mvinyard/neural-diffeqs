
__module_name__ = "_NeuralSDE.py"
__doc__ = """Neural SDE module. Contains API-facing NeuralSDE class."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import packages: --------------------------------------------------------------------
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base import BaseSDE, instantiate_mu_sigma_networks


# -- API-facing class: -------------------------------------------------------------------
class NeuralSDE(BaseSDE):
    def __init__(
        self,
        state_size: int,
        hidden: dict({int: [int, int], int: [int, int]}) = {1: [200, 200]},
        activation_function: 'torch.nn.modules.activation.<func>' = torch.nn.Tanh,
        dropout: float = 0,
        input_bias: bool = True,
        output_bias: bool = True,
        brownian_size: int = 1,
        potential_net: bool = False,
        mu_init: float = None,
        sigma_init: float = None,
        noise_type: str = "general",
        sde_type: str = "ito",
        **kwargs,
    ):
        """
        Instantiate a NeuralSDE.

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

        dropout
            type: float
            default: 0

        potential_net
            If True, overrides out_dim and output_bias, setting them to 1 and
            False, respectively. If potential_net = True, mu_potential_net is 
            set to True, but not sigma_potential_net. For finer control and 
            alternative setups, use the keyword args.
            type: bool
            default: False

        input_bias
            type: bool
            default: True

        output_bias
            type: bool
            default: True

        brownian_size
            type: int
            default: 1

        mu_init
            type: float
            default: None

        sigma_init
            type: float
            default: None

        noise_type
            type: str
            default: "general"

        sde_type
            type: str
            default: "ito"

        kwargs:
        -------
        state_size
            mu_in_dim, mu_out_dim, sigma_in_dim, sigma_out_dim

        hidden
            mu_hidden, sigma_hidden

        activation_function
            mu_activation_function, sigma_activation_function

        input_bias
            mu_input_bias, sigma_input_bias

        output_bias
            mu_output_bias, sigma_output_bias

        dropout
            mu_dropout, sigma_dropout
            
        potential_net
            mu_potential_net, sigma_potential_net

        Keyword arguments are used in sets to replace the simpler, default args. For
        example, if state_size is given, this overrides the more detailed state_size
        keyword arguments, [mu_in_dim, mu_out_dim, sigma_in_dim, sigma_out_dim]. For
        all keyword arguments, the type is consistent with the simpler argument. All
        complex arguments must be passed in order to diverge from using the simpler
        argument.

        Returns:
        --------
        None
        """
        self.__dict__.update(locals())
        super(NeuralSDE, self).__init__()

        self.networks = instantiate_mu_sigma_networks(
            state_size=state_size,
            hidden=hidden,
            activation_function=activation_function,
            dropout=dropout,
            potential_net=potential_net,
            input_bias=input_bias,
            output_bias=output_bias,
            **kwargs,
        )
        self.__setup__()

    # -- drift: --------------------------------------------------------------------------
    def f(self, t: torch.Tensor, y0: torch.Tensor)->torch.Tensor:
        """
        Drift method (term) of the NeuralSDE.

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
        f(y0)
            drift: Tensor of same shape as y0.
            type: torch.Tensor

        Notes:
        ------
        (1) t required syntactically though not functionally.

        Examples:
        ---------
        >>> func = NeuralSDE(state_size)
        >>> x_hat_f = func.f(None, x)
        """
        return self.mu_forward(self.mu, y0)
    
    # -- diffusion: ----------------------------------------------------------------------
    def g(self, t: torch.Tensor, y0: torch.Tensor)->torch.Tensor:
        """
        Diffusion method (term) of the NeuralSDE.

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
        g(y0)
            diffusion: Tensor of same shape as y0.
            type: torch.Tensor

        Notes:
        ------
        (1) t required syntactically though not functionally.
        
        Examples:
        ---------
        >>> func = NeuralSDE(state_size)
        >>> x_hat_g = func.g(None, x)
        """
        return self.view_g_state(self.sigma_forward(self.sigma, y0))

