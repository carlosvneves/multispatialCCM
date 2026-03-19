import numpy as np

from multispatialCCM._rust import get_rust_module
from multispatialCCM.simplex import SSR_pred_boot


def test_missing_rust_module_falls_back_to_python_simplex():
    rust_mod = get_rust_module()
    assert rust_mod is None

    data = np.array([0.1, 0.2, 0.15, 0.22, 0.18, 0.27, 0.21, 0.3], dtype=float)
    out = SSR_pred_boot(A=data, E=2, tau=1, predstep=1)

    assert "rho" in out
    assert np.isfinite(out["rho"])
