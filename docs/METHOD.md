# Method Notes: Multispatial CCM

This package implements multispatial convergent cross mapping (CCM) for replicated short time series, following the original R package and paper.

## Core Idea

Convergent cross mapping (CCM) tests directional causal signal in nonlinear dynamical systems by checking whether information about process `X` is encoded in the reconstructed state space of process `Y`.

Multispatial CCM extends CCM to many short replicated series by concatenating replicated sequences and using bootstrap sampling over valid library points, which allows inference when long single-series records are unavailable.

## Practical Workflow

1. Choose embedding dimensions with `SSR_pred_boot` (usually by maximizing `rho` over candidate `E`).
2. Optionally verify nonlinear signal with `SSR_check_signal`.
3. Run `CCM_boot(A, B, E_A, ...)` and `CCM_boot(B, A, E_B, ...)`.
4. Use `ccmtest` to evaluate whether predictive skill increases from short to long libraries.

## Interpretation

- `pval_a_cause_b`: significance for direction A -> B.
- `pval_b_cause_a`: significance for direction B -> A.

Lower p-values indicate stronger evidence for causal signal in that direction.

## Replication and Numerical Behavior

- Deterministic pieces (for fixed input) should match very closely across backends.
- Bootstrap outputs (`rho`, `sdevrho`) can differ numerically between implementations while preserving the same directional inference.
- For robust inference, prefer larger `iterations` and sensitivity checks across seeds.

## Sources

- CRAN package manual: <https://cran.r-project.org/web/packages/multispatialCCM/multispatialCCM.pdf>
- Ecology (2015) paper (DOI): <https://doi.org/10.1890/14-1479.1>
- Abstract mirror (PubMed): <https://pubmed.ncbi.nlm.nih.gov/26236832/>
