
__module_name__ = "_instantiate_mu_sigma_networks.py"
__doc__ = """Wrangling module to keep the args + kwargs of the NeuralSDE module tidy."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import packages: --------------------------------------------------------------------
import inspect
import licorice_font as lf
import torch_composer


# -- supporting functions: ---------------------------------------------------------------
def is_null(var):
    """Distinct from Zero or False"""
    return isinstance(var, type(None))


def function_kwargs(func):
    return list(inspect.signature(func).parameters)


def mk_dict_func_kwargs(passed_kwargs, func):
    func_kwargs = function_kwargs(func)
    return {k: passed_kwargs.pop(k) for k in dict(passed_kwargs) if k in func_kwargs}


def list_simple_args(args):
    l = list(args.keys())
    l.remove("kwargs")
    return l


class ParseArgsKwargs:
    """Method order matters"""
    def __init__(self):
        pass

        self._warning = lf.font_format("WARNING", ["RED"])

    def __parse_argset__(self, argset: dict):

        """
        if not passing the simple arg, make sure all kwargs are then defined.
        """

        argnames = list(argset.keys())
        simple_arg, complex_argnames = argset[argnames[1]], argnames[2:]
        if not is_null(simple_arg):
            for key in complex_argnames:
                setattr(self, key, simple_arg)
        else:
            missing = []
            for key in complex_argnames:
                if is_null(argset[key]):
                    missing.append(key)
                else:
                    setattr(self, key, argset[key])
            if len(missing) > 0:
                print(" - [ {} ]: Missing kwargs:".format(self._warning), missing)

    def io_state(
        self,
        state_size=None,
        mu_in_dim=None,
        mu_out_dim=None,
        sigma_in_dim=None,
        sigma_out_dim=None,
    ):
        self.__parse_argset__(argset=locals())

    def hidden_state(self, hidden=None, mu_hidden=None, sigma_hidden=None):
        self.__parse_argset__(argset=locals())

    def activation(
        self,
        activation_function=None,
        mu_activation_function=None,
        sigma_activation_function=None,
    ):
        self.__parse_argset__(argset=locals())

    def specify_dropout(self, dropout=None, mu_dropout=None, sigma_dropout=None):
        self.__parse_argset__(argset=locals())

    def potential_net(
        self,
        potential_net=None,
        mu_potential_net=None,
        sigma_potential_net=None,
    ):
        self.__parse_argset__(argset=locals())

    def in_bias(self, input_bias=None, mu_input_bias=None, sigma_input_bias=None):
        self.__parse_argset__(argset=locals())

    def out_bias(self, output_bias=None, mu_output_bias=None, sigma_output_bias=None):
        self.__parse_argset__(argset=locals())


def arg_manager(
    state_size,
    hidden,
    activation_function,
    dropout,
    potential_net,
    input_bias,
    output_bias,
    **kwargs
):

    simple_args = list_simple_args(locals())

    parse = ParseArgsKwargs()

    arg_groups = [attr for attr in parse.__dir__() if not attr.startswith("_")]

    for n, group in enumerate(arg_groups):
        kwargs[simple_args[n]] = locals()[simple_args[n]]
        f = getattr(parse, group)
        f(**mk_dict_func_kwargs(kwargs, f))

    return parse


# -- primary module function : -----------------------------------------------------------
def instantiate_mu_sigma_networks(
    state_size,
    hidden,
    activation_function,
    dropout,
    potential_net,
    input_bias,
    output_bias,
    **kwargs
) -> dict({"str": torch_composer.TorchNet, "str": torch_composer.TorchNet}):

    """
    Takes default args + potential replacement kwargs to instantiate an SDE.

    Parameters:
    -----------
    NeuralSDE params + kwargs
    
    Returns:
    --------
    {"mu": mu_net, "sigma": sigma_net}
        type: dict({"str": torch_composer.TorchNet, "str": torch_composer.TorchNet})
    
    Notes:
    ------
    (1) Purpose: to prevent cluttering of the main API.
    """

    parse = arg_manager(
        state_size=state_size,
        hidden=hidden,
        activation_function=activation_function,
        dropout=dropout,
        potential_net=potential_net,
        input_bias=input_bias,
        output_bias=output_bias,
        **kwargs
    )

    mu_net = torch_composer.TorchNet(
        parse.mu_in_dim,
        parse.mu_out_dim,
        parse.mu_hidden,
        parse.mu_activation_function(),
        parse.mu_dropout,
        parse.mu_input_bias,
        parse.mu_output_bias,
    )

    sigma_net = torch_composer.TorchNet(
        parse.sigma_in_dim,
        parse.sigma_out_dim,
        parse.sigma_hidden,
        parse.sigma_activation_function(),
        parse.sigma_dropout,
        parse.sigma_input_bias,
        parse.sigma_output_bias,
    )

    return {"mu": mu_net, "sigma": sigma_net}