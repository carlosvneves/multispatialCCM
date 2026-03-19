# API Guide

## Data Utilities

- `make_ccm_data()`
- `load_ccm_data(path)`

## Main Analysis Functions

- `SSR_pred_boot(A, B=None, E=2, tau=1, predstep=1, matchSugi=0, backend=None)`
- `SSR_check_signal(A, E, tau, predsteplist)`
- `CCM_boot(A, B, E, tau=1, DesiredL=None, iterations=100, backend=None)`
- `ccmtest(CCM_boot_A, CCM_boot_B)`

## Backend Controls

- `set_backend("auto" | "python" | "rust")`
- `get_backend()`

Per-call `backend` arguments override the global backend setting.

## Minimal Example

```python
import numpy as np
from multispatialCCM import make_ccm_data, SSR_pred_boot, CCM_boot, ccmtest, set_backend

set_backend("auto")

data = make_ccm_data()
A = data["Accm"]
B = data["Bccm"]

Emat = np.zeros((4, 2))
for E in range(2, 6):
    Emat[E - 2, 0] = SSR_pred_boot(A=A, E=E, predstep=1, tau=1)["rho"]
    Emat[E - 2, 1] = SSR_pred_boot(A=B, E=E, predstep=1, tau=1)["rho"]

E_A, E_B = 2, 3
A_causes_B = CCM_boot(A, B, E_A, tau=1, iterations=100)
B_causes_A = CCM_boot(B, A, E_B, tau=1, iterations=100)

print(ccmtest(A_causes_B, B_causes_A))
```
