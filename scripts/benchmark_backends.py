#!/usr/bin/env python3
import argparse
import statistics
import time

from multispatialCCM import CCM_boot, SSR_pred_boot, make_ccm_data


_DEF_E = 3
_DEF_TAU = 1
_DEF_PREDSTEP = 1


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


def run_benchmark(backend: str, repeats: int, ccm_iterations: int):
    data = make_ccm_data()
    a = data["Accm"]
    b = data["Bccm"]

    ssr_times = []
    ccm_times = []

    for _ in range(repeats):
        ssr_times.append(
            _timed_call(
                SSR_pred_boot,
                A=a,
                E=_DEF_E,
                predstep=_DEF_PREDSTEP,
                tau=_DEF_TAU,
                backend=backend,
            )
        )
        ccm_times.append(
            _timed_call(
                CCM_boot,
                A=a,
                B=b,
                E=_DEF_E,
                tau=_DEF_TAU,
                iterations=ccm_iterations,
                backend=backend,
            )
        )

    return {
        "backend": backend,
        "ssr_mean": statistics.mean(ssr_times),
        "ssr_p95": _p95(ssr_times),
        "ccm_mean": statistics.mean(ccm_times),
        "ccm_p95": _p95(ccm_times),
    }


def _print_result(result, repeats, ccm_iterations):
    print(f"backend={result['backend']}")
    print(f"repeats={repeats}, ccm_iterations={ccm_iterations}")
    print(f"SSR_pred_boot: mean={result['ssr_mean']:.6f}s p95={result['ssr_p95']:.6f}s")
    print(f"CCM_boot:      mean={result['ccm_mean']:.6f}s p95={result['ccm_p95']:.6f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark multispatialCCM backends.")
    parser.add_argument("--backend", choices=["python", "rust", "auto", "both"], default="python")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ccm-iterations", type=int, default=30)
    args = parser.parse_args()

    backends = ["python", "rust"] if args.backend == "both" else [args.backend]

    results = [run_benchmark(b, args.repeats, args.ccm_iterations) for b in backends]
    for idx, result in enumerate(results):
        if idx:
            print()
        _print_result(result, args.repeats, args.ccm_iterations)


if __name__ == "__main__":
    main()
