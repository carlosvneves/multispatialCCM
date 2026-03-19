import numpy as np

from multispatialCCM.data import make_ccm_data
from multispatialCCM.simplex import SSR_pred_boot


def test_rust_backend_matches_python_rho_for_simplex():
    data = make_ccm_data()
    a = data["Accm"]

    out_py = SSR_pred_boot(A=a, E=3, tau=1, predstep=1, backend="python")
    out_rust = SSR_pred_boot(A=a, E=3, tau=1, predstep=1, backend="rust")

    assert np.isfinite(out_py["rho"])
    assert np.isfinite(out_rust["rho"])
    assert np.isclose(out_rust["rho"], out_py["rho"], atol=1e-6)
