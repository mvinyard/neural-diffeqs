
__module_name__ = "_base_lightning_NDE.py"
__doc__ = """Base classes for all models."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule


# -- Lightning NDE: ----------------------------------------------------------------------
class BaseLightningNDE(ABC, LightningModule):
    """
    Abstract base class for NeuralDiffEq. Most shared common ancestor class.
    Common to all NDE functions.
    """

    def __init__(self):
        super(BaseLightningDiffEq, self).__init__()
        
    def forward(self):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self, batch, batch_idx):
        pass
