"""
Pure Python implementation of multispatialCCM package.
Translates the R+C implementation to Python with NumPy.
"""

from .data import make_ccm_data, load_ccm_data
from .simplex import SSR_pred_boot
from .ccm import CCM_boot, ccmtest
from .signal import SSR_check_signal

__all__ = [
    "make_ccm_data",
    "SSR_pred_boot",
    "CCM_boot",
    "ccmtest",
    "SSR_check_signal",
]

__version__ = "1.0.0"
