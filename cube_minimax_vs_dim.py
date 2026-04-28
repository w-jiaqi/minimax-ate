import argparse
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_NAMES = [
    "all_signatures",
    "skewed_signatures",
    "normalize_weights",
    "uniform_weights",
    "mou_weights",
    "balanced_weights",
    "get_sizes",
    "build_matrices",
    "solve_minimax",
]


def load_module_from_file(path):
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"minimax file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("user_minimax_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = [name for name in REQUIRED_NAMES if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "loaded module is missing required names: " + ", ".join(missing)
        )
    return module


def choose_design(module, N, m, k):
    if m is not None and k is not None:
        raise ValueError("specify at most one of --m or --k")
    if m is not None:
        w_by_k = module.uniform_weights(N, m)
    elif k is not None:
        w_by_k = module.mou_weights(N, k)
    else:
        w_by_k = module.balanced_weights(N)

    w_by_k = module.normalize_weights(N, w_by_k)
    sizes = list(module.get_sizes(N, w_by_k))
    return w_by_k, sizes


def sample_variance_ddof1(x):
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))


def dim_risk_for_fixed_size(Y, nt):
    N = Y.shape[0]
    nc = N - nt
    if nt <= 0 or nc <= 0:
        raise ValueError("need 1 <= nt <= N-1")

    y0 = Y[:, 0]
    y1 = Y[:, 1]
    tau = y1 - y0

    s1_sq = sample_variance_ddof1(y1)
    s0_sq = sample_variance_ddof1(y0)
    stau_sq = sample_variance_ddof1(tau)

    # Standard Neyman finite-population variance formula.
    return s1_sq / nt + s0_sq / nc - stau_sq / N


def dim_risk_under_design(Y, sizes):
    out = 0.0
    for nt, nc, w in sizes:
        out += w * dim_risk_for_fixed_size(Y, nt)
    return out


def total_number_of_allocations(N, sizes):
    total = 0
    for nt, _nc, w in sizes:
        if w > 0:
            total += math.comb(N, nt)
    return total


def enumerate_allocations_exact(N, sizes):
    # each entry is (treated_indices, prob, nt)
    allocations = []
    for nt, _nc, w in sizes:
        if w <= 0:
            continue
        denom = math.comb(N, nt)
        alloc_prob = w / denom
        for treated in itertools.combinations(range(N), nt):
            allocations.append((np.array(treated, dtype=np.int32), float(alloc_prob), nt))
    return allocations


def sample_allocations_mc(N, sizes, num_allocations, rng):
    nts = np.array([nt for nt, _nc, w in sizes if w > 0], dtype=int)
    ws = np.array([w for nt, _nc, w in sizes if w > 0], dtype=float)
    ws = ws / ws.sum()

    allocations = []
    for _ in range(num_allocations):
        nt = int(rng.choice(nts, p=ws))
        treated = np.sort(rng.choice(N, size=nt, replace=False)).astype(np.int32)
        allocations.append((treated, None, nt))
    return allocations


def poisson_binomial_pmf(probs):
    probs = np.asarray(probs, dtype=float)
    pmf = np.array([1.0], dtype=float)
    for p in probs:
        new = np.zeros(pmf.size + 1, dtype=float)
        new[:-1] += pmf * (1.0 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def extended_minimax_value(Y, allocation, delta_table):
    treated, nt = allocation[0], allocation[2]
    N = Y.shape[0]

    treated_mask = np.zeros(N, dtype=bool)
    treated_mask[treated] = True

    # treated units reveal Y(1), control units reveal Y(0)
    pmf_st = poisson_binomial_pmf(Y[treated_mask, 1])
    pmf_sc = poisson_binomial_pmf(Y[~treated_mask, 0])

    total = 0.0
    for st, p_st in enumerate(pmf_st):
        if p_st == 0.0:
            continue
        for sc, p_sc in enumerate(pmf_sc):
            if p_sc == 0.0:
                continue
            total += delta_table[(nt, st, sc)] * p_st * p_sc
    return float(total)


def sample_unit_cube_states(N, num_states, rng):
    return [rng.uniform(0.0, 1.0, size=(N, 2)) for _ in range(num_states)]


def ate_of_state(Y):
    return float(np.mean(Y[:, 1] - Y[:, 0]))


def estimate_minimax_risk_at_state(Y, allocations, delta_table):
    ate = ate_of_state(Y)

    weighted_sum = 0.0
    use_probs = allocations and allocations[0][1] is not None
    for alloc in allocations:
        _treated, prob, _nt = alloc
        est = extended_minimax_value(Y, alloc, delta_table)
        loss = (est - ate) ** 2
        if use_probs:
            weighted_sum += float(prob) * loss
        else:
            weighted_sum += loss

    if use_probs:
        return float(weighted_sum)
    return float(weighted_sum / len(allocations))


def solve_binary_minimax(module, N, w_by_k, lower, upper,
                         solve_all_binary_states, odd_equivariance, solver):
    out = module.solve_minimax(
        N=N,
        w_by_k=w_by_k,
        lower=lower,
        upper=upper,
        skewed_only=(not solve_all_binary_states),
        odd=odd_equivariance,
        solver=solver,
    )
    delta_table = {stat: float(out["sol"][stat]) for stat in out["grid"]}
    return out, delta_table


def compare_on_random_interior_states(
    module, N, lower, upper, m, k, num_states, num_allocations, seed,
    solve_all_binary_states, odd_equivariance, solver, exact_alloc_threshold
):
    if lower != 0.0 or upper != 1.0:
        raise ValueError(
            "this script is intended for the [0,1] cube. Use --lower 0 --upper 1."
        )

    rng = np.random.default_rng(seed)
    w_by_k, sizes = choose_design(module, N, m, k)
    solve_out, delta_table = solve_binary_minimax(
        module=module,
        N=N,
        w_by_k=w_by_k,
        lower=lower,
        upper=upper,
        solve_all_binary_states=solve_all_binary_states,
        odd_equivariance=odd_equivariance,
        solver=solver,
    )

    total_allocs = total_number_of_allocations(N, sizes)
    use_exact_allocs = total_allocs <= exact_alloc_threshold
    if use_exact_allocs:
        allocations = enumerate_allocations_exact(N, sizes)
    else:
        allocations = sample_allocations_mc(N, sizes, num_allocations, rng)

    states = sample_unit_cube_states(N, num_states, rng)

    rows = []
    mm_better = 0
    dim_better = 0
    ties = 0

    for idx, Y in enumerate(states, start=1):
        ate = ate_of_state(Y)
        dim_risk = dim_risk_under_design(Y, sizes)
        mm_risk = estimate_minimax_risk_at_state(Y, allocations, delta_table)
        diff = dim_risk - mm_risk

        if diff > 1e-12:
            winner = "minimax"
            mm_better += 1
        elif diff < -1e-12:
            winner = "dim"
            dim_better += 1
        else:
            winner = "tie"
            ties += 1

        rows.append({
            "state_id": idx,
            "ate": ate,
            "dim_risk": dim_risk,
            "minimax_risk_est": mm_risk,
            "dim_minus_minimax": diff,
            "winner": winner,
        })

    summary = {
        "N": N,
        "num_states": num_states,
        "seed": seed,
        "lower": lower,
        "upper": upper,
        "design_w_by_k": {int(k0): float(v0) for k0, v0 in w_by_k.items()},
        "solve_binary_states": "all" if solve_all_binary_states else "skewed_only",
        "odd_equivariance": bool(odd_equivariance),
        "solver": solve_out.get("solver"),
        "solver_status": solve_out.get("status"),
        "binary_minimax_worst_risk": float(solve_out.get("worst_risk_orig", np.nan)),
        "allocation_mode": "exact" if use_exact_allocs else "monte_carlo",
        "num_allocations_used": len(allocations),
        "num_exact_allocations_available": total_allocs,
        "states_where_minimax_better": mm_better,
        "states_where_dim_better": dim_better,
        "ties": ties,
        "share_minimax_better": float(mm_better / num_states),
        "avg_dim_risk": float(np.mean([r["dim_risk"] for r in rows])),
        "avg_minimax_risk_est": float(np.mean([r["minimax_risk_est"] for r in rows])),
        "avg_dim_minus_minimax": float(np.mean([r["dim_minus_minimax"] for r in rows])),
    }
    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--minimax-file", type=str, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--upper", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--num-states", type=int, default=50)
    ap.add_argument("--num-allocations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260407)
    ap.add_argument("--solve-all-binary-states", action="store_true")
    ap.add_argument("--no-equivariance", action="store_true")
    ap.add_argument("--solver", type=str, default=None)
    ap.add_argument("--exact-alloc-threshold", type=int, default=5000)
    ap.add_argument("--csv-out", type=str, default=None)
    ap.add_argument("--summary-out", type=str, default=None)

    args = ap.parse_args()

    module = load_module_from_file(args.minimax_file)
    summary, rows = compare_on_random_interior_states(
        module=module,
        N=args.N,
        lower=args.lower,
        upper=args.upper,
        m=args.m,
        k=args.k,
        num_states=args.num_states,
        num_allocations=args.num_allocations,
        seed=args.seed,
        solve_all_binary_states=args.solve_all_binary_states,
        odd_equivariance=(not args.no_equivariance),
        solver=args.solver,
        exact_alloc_threshold=args.exact_alloc_threshold,
    )

    print(f"N={summary['N']}")
    print(f"binary minimax solve over: {summary['solve_binary_states']}")
    print(f"allocation mode: {summary['allocation_mode']}")
    print(f"allocations used: {summary['num_allocations_used']}")
    print(f"random states sampled: {summary['num_states']}")
    print(f"states where minimax better: {summary['states_where_minimax_better']}")
    print(f"states where DiM better: {summary['states_where_dim_better']}")
    print(f"ties: {summary['ties']}")
    print(f"share minimax better: {summary['share_minimax_better']:.4f}")
    print(f"avg DiM risk: {summary['avg_dim_risk']:.6g}")
    print(f"avg minimax risk estimate: {summary['avg_minimax_risk_est']:.6g}")
    print(f"avg (DiM - minimax): {summary['avg_dim_minus_minimax']:.6g}")
    print(f"binary minimax worst-case risk: {summary['binary_minimax_worst_risk']:.6g}")

    if args.csv_out:
        write_csv(args.csv_out, rows)
        print(f"wrote per-state CSV to {args.csv_out}")
    if args.summary_out:
        write_json(args.summary_out, summary)
        print(f"wrote summary JSON to {args.summary_out}")


if __name__ == "__main__":
    main()
