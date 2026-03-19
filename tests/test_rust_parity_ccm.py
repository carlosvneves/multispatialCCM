import numpy as np

from multispatialCCM.data import make_ccm_data
from multispatialCCM.ccm import CCM_boot, ccmtest


def test_rust_backend_matches_python_rho_for_ccm():
    data = make_ccm_data()
    a = data["Accm"]
    b = data["Bccm"]

    ba_py = CCM_boot(a, b, E=3, tau=1, iterations=30, backend="python")
    ab_py = CCM_boot(b, a, E=3, tau=1, iterations=30, backend="python")
    ba_rust = CCM_boot(a, b, E=3, tau=1, iterations=30, backend="rust")
    ab_rust = CCM_boot(b, a, E=3, tau=1, iterations=30, backend="rust")

    assert ba_py["rho"].shape == ba_rust["rho"].shape
    assert ab_py["rho"].shape == ab_rust["rho"].shape
    assert np.isfinite(ba_rust["rho"]).all()
    assert np.isfinite(ab_rust["rho"]).all()

    sig_py = ccmtest(ba_py, ab_py)
    sig_rust = ccmtest(ba_rust, ab_rust)

    assert abs(sig_py["pval_a_cause_b"] - sig_rust["pval_a_cause_b"]) <= 0.25
    assert abs(sig_py["pval_b_cause_a"] - sig_rust["pval_b_cause_a"]) <= 0.15
