
__module_name__ = "_base_support.py"
__doc__ = """Functions to support the base classes."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard@broadinstitute.org"])


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.2"


# -- import packages: --------------------------------------------------------------------
import torch_composer


# -- supporting base functions: ----------------------------------------------------------
as_list = torch_composer._core.base.as_list


def set_state_sizes(state_size, inferred_state_size):
    if not inferred_state_size:
        return state_size
    return inferred_state_size

def is_potential_net(net):
    return list(net.parameters())[-1].data.numel() == 1

def define_forward_function(self, name):
    if is_potential_net(getattr(self, name)):
        setattr(self, "{}_forward".format(name), getattr(self, "potential_pass"))
        setattr(self, "is_{}_potential".format(name), True)
    else:
        setattr(self, "{}_forward".format(name), getattr(self, "forward_pass"))
        setattr(self, "is_{}_potential".format(name), False)

def parse_passed_networks(self, networks: dict):

    for name, network in networks.items():
        setattr(self, name, network)
