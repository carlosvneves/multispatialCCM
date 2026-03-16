#!/usr/bin/env python3
"""
DoD Validation Test — two scenarios:
  Scenario 1: Full pipeline with Python-generated data (make_ccm_data)
  Scenario 2: R-generated data (ccm_data.csv) with Python CCM → must match DoD exactly
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/carloseduardoverasneves/Downloads/multispatialCCM")

from multispatialCCM import (
    make_ccm_data,
    load_ccm_data,
    SSR_pred_boot,
    CCM_boot,
    ccmtest,
)

PASS = "PASS"
FAIL = "FAIL"


def run_ssr_emat(Accm, Bccm, maxE=5, label=""):
    Emat = np.zeros((maxE - 1, 2))
    for E in range(2, maxE + 1):
        rA = SSR_pred_boot(A=Accm, E=E, predstep=1, tau=1)
        rB = SSR_pred_boot(A=Bccm, E=E, predstep=1, tau=1)
        Emat[E - 2, 0] = rA["rho"] if np.isfinite(rA["rho"]) else 0.0
        Emat[E - 2, 1] = rB["rho"] if np.isfinite(rB["rho"]) else 0.0
    return Emat


def run_ccm(Accm, Bccm, E_A, E_B, iterations=100):
    CCM_boot_A = CCM_boot(Accm, Bccm, E_A, tau=1, iterations=iterations)
    CCM_boot_B = CCM_boot(Bccm, Accm, E_B, tau=1, iterations=iterations)
    sig = ccmtest(CCM_boot_A, CCM_boot_B)
    return CCM_boot_A, CCM_boot_B, sig


def plot_emat(Emat, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    Es = list(range(2, 2 + len(Emat)))
    ax.plot(Es, Emat[:, 0], color="black", lw=2, label="A")
    ax.plot(Es, Emat[:, 1], color="#c0396b", lw=2, ls="--", label="B")
    ax.set_xlabel("E")
    ax.set_ylabel("rho")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   Saved: {out_path}")


def plot_ccm(CCM_boot_A, CCM_boot_B, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    xlim = (
        min(CCM_boot_A["Lobs"].min(), CCM_boot_B["Lobs"].min()),
        max(CCM_boot_A["Lobs"].max(), CCM_boot_B["Lobs"].max()),
    )

    # A causes B (solid black)
    ax.plot(CCM_boot_A["Lobs"], CCM_boot_A["rho"], color="black", lw=2, label="A causes B")
    ax.plot(CCM_boot_A["Lobs"], CCM_boot_A["rho"] - CCM_boot_A["sdevrho"],
            color="black", lw=1, ls=":")
    ax.plot(CCM_boot_A["Lobs"], CCM_boot_A["rho"] + CCM_boot_A["sdevrho"],
            color="black", lw=1, ls=":")

    # B causes A (dashed red)
    ax.plot(CCM_boot_B["Lobs"], CCM_boot_B["rho"], color="#c0396b", lw=2, ls="--", label="B causes A")
    ax.plot(CCM_boot_B["Lobs"], CCM_boot_B["rho"] - CCM_boot_B["sdevrho"],
            color="#c0396b", lw=1, ls=":")
    ax.plot(CCM_boot_B["Lobs"], CCM_boot_B["rho"] + CCM_boot_B["sdevrho"],
            color="#c0396b", lw=1, ls=":")

    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_xlabel("L")
    ax.set_ylabel("rho")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Full pipeline with Python-generated data
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SCENARIO 1: Full Python pipeline (Python-generated data)")
print("=" * 65)

ccm_data_py = make_ccm_data()
Accm_py = ccm_data_py["Accm"]
Bccm_py = ccm_data_py["Bccm"]
print(f"  Data generated: len={len(Accm_py)}")

print("  Computing Emat...")
Emat_py = run_ssr_emat(Accm_py, Bccm_py, label="Python-gen")
print("  E     A        B")
for i, E in enumerate(range(2, 6)):
    print(f"  {E}    {Emat_py[i,0]:.4f}   {Emat_py[i,1]:.4f}")

E_A_py = int(np.argmax(Emat_py[:, 0])) + 2
E_B_py = int(np.argmax(Emat_py[:, 1])) + 2
print(f"  Best E_A={E_A_py}, E_B={E_B_py}")

print("  Running CCM (100 iterations)...")
CBA_py, CBB_py, sig_py = run_ccm(Accm_py, Bccm_py, E_A_py, E_B_py, iterations=100)

pval_ab_py = sig_py["pval_a_cause_b"]
pval_ba_py = sig_py["pval_b_cause_a"]
print(f"  pval_a_cause_b = {pval_ab_py:.2f}")
print(f"  pval_b_cause_a = {pval_ba_py:.2f}")

# Directional correctness check (B causes A, not A causes B)
ok1 = pval_ba_py < 0.05 and pval_ab_py >= 0.05
print(f"  Direction check (B→A sig, A→B not): [{PASS if ok1 else FAIL}]")

plot_emat(Emat_py,  "Scenario 1 — E selection (Python data)", "output_E_py.png")
plot_ccm(CBA_py, CBB_py, "Scenario 1 — CCM (Python data)", "output_rho_py.png")


# ─────────────────────────────────────────────────────────────
# SCENARIO 2: R-generated data → Python CCM → compare to DoD
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SCENARIO 2: R-generated data + Python CCM (DoD comparison)")
print("=" * 65)

ccm_data_r = load_ccm_data()
Accm_r = ccm_data_r["Accm"]
Bccm_r = ccm_data_r["Bccm"]
print(f"  Data loaded from R CSV: len={len(Accm_r)}")

# DoD expected Emat values
EXPECTED_EMAT = np.array([
    [0.4109265, 0.4985892],
    [0.3872952, 0.5589368],
    [0.3696862, 0.5338485],
    [0.2503559, 0.4702056],
])
TOL_EMAT = 1e-4

print("  Computing Emat...")
Emat_r = run_ssr_emat(Accm_r, Bccm_r, label="R-data")
print("  E     A (Python)    A (R-DoD)  |diff|    B (Python)    B (R-DoD)  |diff|")
emat_ok = True
for i, E in enumerate(range(2, 6)):
    dA = abs(Emat_r[i, 0] - EXPECTED_EMAT[i, 0])
    dB = abs(Emat_r[i, 1] - EXPECTED_EMAT[i, 1])
    ok = dA <= TOL_EMAT and dB <= TOL_EMAT
    if not ok:
        emat_ok = False
    mark = PASS if ok else FAIL
    print(f"  {E}   {Emat_r[i,0]:.7f}   {EXPECTED_EMAT[i,0]:.7f}   {dA:.2e}  "
          f"  {Emat_r[i,1]:.7f}   {EXPECTED_EMAT[i,1]:.7f}   {dB:.2e}   [{mark}]")

print(f"  Emat match (tol={TOL_EMAT}): [{PASS if emat_ok else FAIL}]")

E_A_r = 2
E_B_r = 3
print(f"  Using DoD E_A={E_A_r}, E_B={E_B_r}")

print("  Running CCM (100 iterations)...")
CBA_r, CBB_r, sig_r = run_ccm(Accm_r, Bccm_r, E_A_r, E_B_r, iterations=100)

pval_ab_r = sig_r["pval_a_cause_b"]
pval_ba_r = sig_r["pval_b_cause_a"]
print(f"  pval_a_cause_b = {pval_ab_r:.2f}  (DoD: 0.2)")
print(f"  pval_b_cause_a = {pval_ba_r:.2f}  (DoD: 0.0)")

# p-values must match DoD to 1 decimal place
ok_pval_ab = abs(pval_ab_r - 0.2) < 0.05
ok_pval_ba = pval_ba_r == 0.0
print(f"  pval_a_cause_b match: [{PASS if ok_pval_ab else FAIL}]")
print(f"  pval_b_cause_a match: [{PASS if ok_pval_ba else FAIL}]")

# CCM convergence shape check
# A causes B: rho must remain flat (max - min < 0.15 and max < 0.25)
rho_ab_range = float(np.nanmax(CBA_r["rho"]) - np.nanmin(CBA_r["rho"]))
rho_ab_max   = float(np.nanmax(CBA_r["rho"]))
ok_flat = rho_ab_range < 0.25 and rho_ab_max < 0.30
# B causes A: rho must increase substantially (max > 0.45)
rho_ba_max = float(np.nanmax(CBB_r["rho"]))
ok_conv = rho_ba_max > 0.45
print(f"  A→B rho flat (max={rho_ab_max:.3f}, range={rho_ab_range:.3f}): [{PASS if ok_flat else FAIL}]")
print(f"  B→A rho converges (max={rho_ba_max:.3f}): [{PASS if ok_conv else FAIL}]")

plot_emat(Emat_r,  "Scenario 2 — E selection (R data)", "output_E_r.png")
plot_ccm(CBA_r, CBB_r, "Scenario 2 — CCM (R data) vs DoD", "output_rho_r.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
all_pass = ok1 and emat_ok and ok_pval_ab and ok_pval_ba and ok_flat and ok_conv
checks = {
    "Scenario 1 — directional CCM correctness": ok1,
    "Scenario 2 — Emat matches R (tol=1e-4)": emat_ok,
    "Scenario 2 — pval_a_cause_b ≈ 0.2": ok_pval_ab,
    "Scenario 2 — pval_b_cause_a = 0.0": ok_pval_ba,
    "Scenario 2 — A→B rho flat": ok_flat,
    "Scenario 2 — B→A rho converges": ok_conv,
}
for desc, result in checks.items():
    print(f"  [{PASS if result else FAIL}] {desc}")
print()
print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
print("=" * 65)

sys.exit(0 if all_pass else 1)
