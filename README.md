# multispatialCCM

Pure-Python translation of the R package [`multispatialCCM`](https://cran.r-project.org/package=multispatialCCM) (Clark et al. 2015).

Implements **Convergent Cross Mapping (CCM)** for detecting causal relationships in time series from spatially replicated systems, based on Sugihara et al. (2012).

## Installation

```bash
pip install .
# or, with uv:
uv add .
```

### Install From GitHub

```bash
pip install "git+https://github.com/carlosvneves/multispatialCCM.git@main"
# or, with uv:
uv pip install "git+https://github.com/carlosvneves/multispatialCCM.git@main"
```

Need compatibility with the previous implementation?

```bash
uv pip install "git+https://github.com/carlosvneves/multispatialCCM.git@legacy"
```

## Quick start

```python
from multispatialCCM import make_ccm_data, SSR_pred_boot, CCM_boot, ccmtest
import numpy as np

# Generate simulated data (B causes A, not the reverse)
data = make_ccm_data()
Accm = data["Accm"]
Bccm = data["Bccm"]

# 1. Find optimal embedding dimension E
maxE = 5
Emat = np.zeros((maxE - 1, 2))
for E in range(2, maxE + 1):
    Emat[E - 2, 0] = SSR_pred_boot(A=Accm, E=E, predstep=1, tau=1)["rho"]
    Emat[E - 2, 1] = SSR_pred_boot(A=Bccm, E=E, predstep=1, tau=1)["rho"]

E_A = int(np.argmax(Emat[:, 0])) + 2  # E where rho(A) is maximised
E_B = int(np.argmax(Emat[:, 1])) + 2  # E where rho(B) is maximised

# 2. Run CCM
CCM_boot_A = CCM_boot(Accm, Bccm, E_A, tau=1, iterations=100)
CCM_boot_B = CCM_boot(Bccm, Accm, E_B, tau=1, iterations=100)

# 3. Test significance
result = ccmtest(CCM_boot_A, CCM_boot_B)
print(result)
# {'pval_a_cause_b': 0.2, 'pval_b_cause_a': 0.0}
# → B causes A (significant); A does not cause B
```

## Backend Selection

`SSR_pred_boot` and `CCM_boot` accept `backend`:
- `"auto"` (default): tries native Rust path when available, otherwise Python.
- `"python"`: always use Python path.
- `"rust"`: prefer Rust path, fallback to Python if native module is unavailable.

You can also set a global default:

```python
from multispatialCCM import set_backend, get_backend

set_backend("rust")  # "auto" | "python" | "rust"
print(get_backend())
```


## Documentation

- Method: [docs/METHOD.md](docs/METHOD.md)
- API: [docs/API.md](docs/API.md)
- Installation and packaging: [docs/INSTALL.md](docs/INSTALL.md)

## API

| Function | Description |
|---|---|
| `SSR_pred_boot(A, B, E, tau, predstep)` | Simplex projection — finds optimal embedding dimension |
| `SSR_check_signal(A, E, tau, predsteplist)` | Tests for nonlinear signal (rho decline with predstep) |
| `CCM_boot(A, B, E, tau, iterations)` | CCM with bootstrap — tests causal direction |
| `ccmtest(CCM_boot_A, CCM_boot_B)` | Returns p-values for both causal directions |
| `make_ccm_data()` | Generates simulated coupled logistic map data |
| `load_ccm_data(path)` | Loads CCM data from a CSV file |
| `set_backend(mode)` / `get_backend()` | Configure or inspect global backend mode |

## Validation

Running `test_dod.py` reproduces the reference results from the original R package:

```
Emat (R-generated data):
  E=2: A=0.4109  B=0.4986
  E=3: A=0.3873  B=0.5589
  E=4: A=0.3697  B=0.5338
  E=5: A=0.2504  B=0.4702

CCM significance test:
  pval_a_cause_b = 0.2   (A does not cause B)
  pval_b_cause_a = 0.0   (B causes A ✓)
```

Emat values match R to < 1e-6. CCM p-values match the DoD.

## Notes

- `make_ccm_data()` uses Python's RNG, so it produces different random data than R's `make_ccm_data()`. The causal structure (B→A) is preserved.
- For exact R-equivalent data, generate it with R and load via `load_ccm_data(path)`.

## References

- Clark, A.T. et al. (2015). *Spatial convergent cross mapping to detect causal relationships from short time series.* Ecology, 96(5), 1174–1181.
- Sugihara, G. et al. (2012). *Detecting causality in complex ecosystems.* Science, 338(6106), 496–500.
