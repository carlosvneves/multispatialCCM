import importlib
import numpy as np

import multispatialCCM._rust as rust_loader
from multispatialCCM.simplex import SSR_pred_boot


def test_missing_rust_module_falls_back_to_python_simplex(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import(name, package=None):
        if name in {"multispatialCCM._multispatialccm_rust", "_multispatialccm_rust"}:
            raise ImportError("forced for fallback test")
        return real_import_module(name, package)

    monkeypatch.setattr(rust_loader, "import_module", fake_import)

    rust_mod = rust_loader.get_rust_module()
    assert rust_mod is None

    data = np.array([0.1, 0.2, 0.15, 0.22, 0.18, 0.27, 0.21, 0.3], dtype=float)
    out = SSR_pred_boot(A=data, E=2, tau=1, predstep=1, backend="auto")

    assert "rho" in out
    assert np.isfinite(out["rho"])
