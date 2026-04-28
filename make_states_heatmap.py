import argparse
import csv
import importlib.util
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
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
    x = np.asarray(x, dtype=float)
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

    return s1_sq / nt + s0_sq / nc - stau_sq / N


def dim_risk_under_design(Y, sizes):
    out = 0.0
    for nt, _nc, w in sizes:
        out += w * dim_risk_for_fixed_size(Y, nt)
    return out


def total_number_of_allocations(N, sizes):
    total = 0
    for nt, _nc, w in sizes:
        if w > 0:
            total += math.comb(N, nt)
    return total


def enumerate_allocations_exact(N, sizes):
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
    treated, _prob, nt = allocation
    N = Y.shape[0]

    treated_mask = np.zeros(N, dtype=bool)
    treated_mask[treated] = True

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


def ate_of_state(Y):
    return float(np.mean(Y[:, 1] - Y[:, 0]))


def heterogeneity_of_state(Y):
    v0 = float(np.var(Y[:, 0], ddof=0))
    v1 = float(np.var(Y[:, 1], ddof=0))
    return math.sqrt(v0 + v1)


def estimate_minimax_risk_at_state(Y, allocations, delta_table):
    ate = ate_of_state(Y)
    weighted_sum = 0.0
    use_probs = allocations and allocations[0][1] is not None

    for alloc in allocations:
        est = extended_minimax_value(Y, alloc, delta_table)
        loss = (est - ate) ** 2
        if use_probs:
            weighted_sum += float(alloc[1]) * loss
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

#
# def parse_sampler_specs(spec_text, default_sigma=0.05):
#     if spec_text is None or spec_text.strip() == "":
#         return [("uniform_cube", None, None)]
#
#     specs = []
#     parts = [p.strip() for p in spec_text.split(",") if p.strip()]
#     for part in parts:
#         toks = [t.strip() for t in part.split(":")]
#
#         if toks[0] in ("uniform_cube", "homogeneous"):
#             if len(toks) != 2:
#                 raise ValueError(f"bad sampler spec: {part}")
#             specs.append((toks[0], default_sigma, int(toks[1])))
#
#         elif toks[0] == "smooth_normal":
#             if len(toks) == 2:
#                 specs.append((toks[0], float(default_sigma), int(toks[1])))
#             elif len(toks) == 3:
#                 specs.append((toks[0], float(toks[1]), int(toks[2])))
#             else:
#                 raise ValueError(f"bad sampler spec: {part}")
#
#         else:
#             raise ValueError(f"unknown sampler in spec: {part}")
#
#     return specs
#
#
# def sample_states_from_specs(N, rng, sampler_specs):
#     states = []
#     for sampler, sigma, count in sampler_specs:
#         if sampler == "uniform_cube":
#             for _ in range(count):
#                 Y = rng.uniform(0.0, 1.0, size=(N, 2))
#                 states.append((sampler, None, Y))
#         elif sampler == "homogeneous":
#             for _ in range(count):
#                 base = rng.uniform(0.0, 1.0, size=2)
#                 Y = np.tile(base.reshape(1, 2), (N, 1))
#                 states.append((sampler, None, Y))
#         elif sampler == "smooth_normal":
#             for _ in range(count):
#                 base = rng.uniform(0.0, 1.0, size=2)
#                 Y = np.empty((N, 2), dtype=float)
#                 Y[:, 0] = base[0] + rng.normal(0.0, sigma, size=N)
#                 Y[:, 1] = base[1] + rng.normal(0.0, sigma, size=N)
#                 Y = np.clip(Y, 0.0, 1.0)
#                 states.append((sampler, sigma, Y))
#         else:
#             raise ValueError(f"unknown sampler: {sampler}")
#     return states


def sample_unit_cube_states(N, num_states, rng):
    return [rng.uniform(0.0, 1.0, size=(N, 2)) for _ in range(num_states)]


def compute_rows(module, N, lower, upper, m, k, num_states, num_allocations, seed,
                 solve_all_binary_states, odd_equivariance, solver, exact_alloc_threshold):
    #sampler_specs parameter replaces num_states
    # def compute_rows(module, N, lower, upper, m, k, num_allocations, seed,
    #                  solve_all_binary_states, odd_equivariance, solver, exact_alloc_threshold,
    #                  sampler_specs):
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
    #states = sample_states_from_specs(N, rng, sampler_specs)

    rows = []
    mm_better = 0
    dim_better = 0
    ties = 0

    for idx, Y in enumerate(states, start=1):
        #for idx, (sampler_name, sigma, Y) in enumerate(states, start=1):
        ate = ate_of_state(Y)
        heter = heterogeneity_of_state(Y)
        dim_risk = dim_risk_under_design(Y, sizes)
        mm_risk = estimate_minimax_risk_at_state(Y, allocations, delta_table)
        gap = dim_risk - mm_risk

        if gap > 1e-12:
            winner = "minimax"
            mm_better += 1
        elif gap < -1e-12:
            winner = "dim"
            dim_better += 1
        else:
            winner = "tie"
            ties += 1

        rows.append({
            "state_id": idx,
            "sampler": "uniform_cube",
            #"sampler": sampler_name,
            #"sigma": "" if sigma is None else sigma,
            "ate": ate,
            "abs_ate": abs(ate),
            "heter": heter,
            "dim_risk": dim_risk,
            "mm_risk": mm_risk,
            "gap": gap,
            "winner": winner,
        })

    avg_gap = sum(r["gap"] for r in rows) / len(rows)

    summary = {
        "N": N,
        "num_states": num_states,
        #"num_states": len(rows),
        #"sampler_specs": sampler_specs,
        "seed": seed,
        "solve_binary_states": "all" if solve_all_binary_states else "skewed_only",
        "allocation_mode": "exact" if use_exact_allocs else "monte_carlo",
        "num_allocations_used": len(allocations),
        "states_where_minimax_better": mm_better,
        "states_where_dim_better": dim_better,
        "ties": ties,
        "share_minimax_better": float(mm_better / num_states),
        #"share_minimax_better": float(mm_better / len(rows)),
        "avg_gap": avg_gap,
        "binary_minimax_worst_risk": float(solve_out.get("worst_risk_orig", np.nan)),
    }
    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_heatmap(rows, plot_out, x_max=0.55, y_max=0.60, x_bins=12, y_bins=12, vlim=0.01):
    xedges = np.linspace(0.0, x_max, x_bins + 1)
    yedges = np.linspace(0.0, y_max, y_bins + 1)
    sums = np.zeros((y_bins, x_bins), dtype=float)
    counts = np.zeros((y_bins, x_bins), dtype=int)

    for r in rows:
        x = float(r["heter"])
        y = float(r["abs_ate"])
        z = float(r["gap"])
        ix = np.searchsorted(xedges, x, side="right") - 1
        iy = np.searchsorted(yedges, y, side="right") - 1
        if 0 <= ix < x_bins and 0 <= iy < y_bins:
            sums[iy, ix] += z
            counts[iy, ix] += 1

    grid = np.full((y_bins, x_bins), np.nan, dtype=float)
    mask = counts > 0
    grid[mask] = sums[mask] / counts[mask]
    masked = np.ma.masked_invalid(grid)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=180)
    im = ax.imshow(
        masked,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
    )
    ax.set_xlabel("Heterogeneity H")
    ax.set_ylabel("|ATE|")
    ax.set_title("Average risk gap: DiM - minimax\n(red = minimax better, blue = DiM better)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean gap")
    fig.tight_layout()
    fig.savefig(plot_out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--minimax-file", type=str, required=True)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--upper", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--num-states", type=int, default=600)
    #ap.add_argument(
    #     "--samplers", type=str, default=None,
    #     help="comma-separated sampler specs, e.g. "
    #          "'uniform_cube:1000,smooth_normal:0.05:100,homogeneous:50'"
    # )
    #ap.add_argument(
    #     "--sigma", type=float, default=0.05,
    #     help="default sigma for smooth_normal if not given explicitly"
    # )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solve-all-binary-states", action="store_true")
    ap.add_argument("--no-equivariance", action="store_true")
    ap.add_argument("--solver", type=str, default=None)
    ap.add_argument("--exact-alloc-threshold", type=int, default=5000)
    ap.add_argument("--num-allocations", type=int, default=50)
    ap.add_argument("--csv-out", type=str, default=None)
    ap.add_argument("--plot-out", type=str, default="left_heatmap_only.png")

    args = ap.parse_args()

    module = load_module_from_file(args.minimax_file)

    # if args.samplers is None:
    #     sampler_specs = [("uniform_cube", args.sigma, args.num_states)]
    # else:
    #     sampler_specs = parse_sampler_specs(args.samplers, default_sigma=args.sigma)

    summary, rows = compute_rows(
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
        #sampler_specs=sampler_specs,
        #num_states removed from above
    )

    make_heatmap(rows, args.plot_out)

    print(f"N={summary['N']}")
    print("sampler: uniform_cube")
    #print(f"samplers: {summary['sampler_specs']}")
    print(f"binary minimax solve over: {summary['solve_binary_states']}")
    print(f"allocation mode: {summary['allocation_mode']}")
    print(f"allocations used: {summary['num_allocations_used']}")
    print(f"random states sampled: {summary['num_states']}")
    print(f"states where minimax better: {summary['states_where_minimax_better']}")
    print(f"states where DiM better: {summary['states_where_dim_better']}")
    print(f"ties: {summary['ties']}")
    print(f"share minimax better: {summary['share_minimax_better']:.4f}")
    print(f"avg gap (DiM - minimax): {summary['avg_gap']:.6g}")
    print(f"wrote heatmap to {args.plot_out}")
    if args.csv_out:
        write_csv(args.csv_out, rows)
        print(f"wrote per-state CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
