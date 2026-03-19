import numpy as np

from multispatialCCM.data import make_ccm_data
from multispatialCCM.ccm import CCM_boot


def test_rust_backend_matches_python_rho_for_ccm():
    data = make_ccm_data()
    a = data["Accm"]
    b = data["Bccm"]

    out_py = CCM_boot(a, b, E=3, tau=1, iterations=10, backend="python")
    out_rust = CCM_boot(a, b, E=3, tau=1, iterations=10, backend="rust")

    assert out_py["rho"].shape == out_rust["rho"].shape
    assert np.allclose(out_rust["rho"], out_py["rho"], atol=1e-6, equal_nan=True)
