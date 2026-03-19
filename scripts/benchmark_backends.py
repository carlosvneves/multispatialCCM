#!/usr/bin/env python3
import argparse
import statistics
import time

from multispatialCCM import CCM_boot, SSR_pred_boot, make_ccm_data


def _timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def _p95(values):
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def run_python_benchmark(repeats: int, ccm_iterations: int):
    data = make_ccm_data()
    A = data["Accm"]
    B = data["Bccm"]

    ssr_times = []
    ccm_times = []

    for _ in range(repeats):
        ssr_times.append(_timed_call(SSR_pred_boot, A=A, E=3, predstep=1, tau=1))
        ccm_times.append(
            _timed_call(CCM_boot, A=A, B=B, E=3, tau=1, iterations=ccm_iterations)
        )

    return {
        "ssr_mean": statistics.mean(ssr_times),
        "ssr_p95": _p95(ssr_times),
        "ccm_mean": statistics.mean(ccm_times),
        "ccm_p95": _p95(ccm_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multispatialCCM backends (baseline: Python)."
    )
    parser.add_argument("--backend", choices=["python"], default="python")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ccm-iterations", type=int, default=30)
    args = parser.parse_args()

    result = run_python_benchmark(args.repeats, args.ccm_iterations)

    print("backend=python")
    print(f"repeats={args.repeats}, ccm_iterations={args.ccm_iterations}")
    print(f"SSR_pred_boot: mean={result['ssr_mean']:.6f}s p95={result['ssr_p95']:.6f}s")
    print(f"CCM_boot:      mean={result['ccm_mean']:.6f}s p95={result['ccm_p95']:.6f}s")


if __name__ == "__main__":
    main()
