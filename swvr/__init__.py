"""SWVR: ``softmax(h @ W.T + b) @ E`` (and an MSE loss on it) without materializing ``[N, V]``."""
from .api import SoftmaxReadoutMSE, softmax_readout, softmax_readout_mse
from .reference import softmax_readout_reference

__all__ = ["softmax_readout", "softmax_readout_mse", "SoftmaxReadoutMSE", "softmax_readout_reference"]
__version__ = "0.1.0"
