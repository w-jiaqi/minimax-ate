import argparse
import math

import numpy as np
import cvxpy as cp


def logC(n, k):
    """Log of C(n, k) using lgamma — stable for large n."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def all_signatures(N):
    sigs = []
    for n00 in range(N + 1):
        for n01 in range(N - n00 + 1):
            for n10 in range(N - n00 - n01 + 1):
                n11 = N - n00 - n01 - n10
                sigs.append((n00, n01, n10, n11))
    return sigs


def skewed_signatures(N):
    sigs = set()

    for n00 in range(N + 1):
        for n11 in range(N - n00 + 1):
            n01 = N - n00 - n11
            sigs.add((n00, n01, 0, n11))

    for n00 in range(N + 1):
        for n11 in range(N - n00 + 1):
            n10 = N - n00 - n11
            sigs.add((n00, 0, n10, n11))

    return sorted(sigs)


def normalize_weights(N, w_by_k):
    for k in w_by_k:
        if k < 1 or k > N - 1:
            raise ValueError("k must be in 1..N-1")
        if w_by_k[k] < 0:
            raise ValueError("weights must be >= 0")

    total = sum(w_by_k.values())
    if total <= 0:
        raise ValueError("weights must sum to > 0")

    return {k: w_by_k[k] / total for k in w_by_k}


def check_symmetry(N, w_by_k):
    for k in w_by_k:
        wk = w_by_k[k]
        wk2 = w_by_k.get(N - k, 0.0)
        if abs(wk - wk2) > 1e-12:
            raise ValueError(
                f"Design not symmetric: w_{k}={wk} but w_{N-k}={wk2}")


def uniform_weights(N, m):
    if m < 1 or m > N - 1:
        raise ValueError("m must be in 1..N-1")
    return {m: 1.0}


def mou_weights(N, k):
    if N % 2 == 0:
        m1 = N // 2 + k
        m2 = N // 2 - k
    else:
        m1 = N // 2 + k + 1
        m2 = N // 2 - k

    if m1 < 1 or m1 > N - 1 or m2 < 1 or m2 > N - 1:
        raise ValueError(f"k={k} gives invalid treatment sizes for N={N}")

    if m1 == m2:
        return {m1: 1.0}
    return {m1: 0.5, m2: 0.5}


def balanced_weights(N):
    return mou_weights(N, 0)


def get_sizes(N, w_by_k):
    w_by_k = normalize_weights(N, w_by_k)
    sizes = []
    for k in sorted(w_by_k):
        sizes.append((k, N - k, w_by_k[k]))
    return sizes


def signature_theta(sig, N):
    n00, n01, n10, n11 = sig
    return (n01 - n10) / N


# ---------------------------------------------------------------------------
# Grid and PMF over observable statistics (nt, st, sc)
# ---------------------------------------------------------------------------

def stat_grid(N, sizes):
    """
    Build the set of observable statistics (nt, st, sc).
    Returns sorted list of tuples and an index dict.
    """
    grid = set()
    for nt, nc, w in sizes:
        if w <= 0:
            continue
        for st in range(nt + 1):
            for sc in range(nc + 1):
                grid.add((nt, st, sc))
    grid = sorted(grid)
    grid_index = {g: i for i, g in enumerate(grid)}
    return grid, grid_index


def compute_pmf(sig, N, sizes, grid_index):
    """
    For a signature (n00, n01, n10, n11), compute:
      theta = (n01 - n10)/N
      p[g]  = P(observe statistic g) under the randomization design
    Uses log-space arithmetic to avoid overflow for large N.
    """
    n00, n01, n10, n11 = sig
    theta = signature_theta(sig, N)

    G = len(grid_index)
    p = np.zeros(G)

    for nt, nc, w in sizes:
        if w <= 0:
            continue

        log_denom = logC(N, nt)
        log_w = math.log(w)

        for x01 in range(min(n01, nt) + 1):
            for x10 in range(min(n10, nt - x01) + 1):
                for x11 in range(min(n11, nt - x01 - x10) + 1):
                    x00 = nt - x01 - x10 - x11
                    if x00 < 0 or x00 > n00:
                        continue

                    log_num = (logC(n00, x00) + logC(n01, x01) +
                               logC(n10, x10) + logC(n11, x11))
                    prob = math.exp(log_w + log_num - log_denom)

                    st = x01 + x11
                    sc = (n10 - x10) + (n11 - x11)

                    p[grid_index[(nt, st, sc)]] += prob

    return theta, p


def build_matrices(N, sizes, sigs):
    """Build the grid, theta vector, and PMF matrix P."""
    grid, grid_index = stat_grid(N, sizes)

    S = len(sigs)
    G = len(grid)

    theta_vec = np.zeros(S)
    P = np.zeros((S, G))

    for i, sig in enumerate(sigs):
        theta, p = compute_pmf(sig, N, sizes, grid_index)
        theta_vec[i] = theta
        P[i, :] = p

    row_sums = P.sum(axis=1)
    bad = np.abs(row_sums - 1.0) > 1e-6
    if bad.any():
        idx = int(np.where(bad)[0][0])
        raise RuntimeError(
            f"PMF row {idx} (sig={sigs[idx]}) sums to "
            f"{row_sums[idx]:.8g}, expected 1")

    return grid, grid_index, theta_vec, P


# ---------------------------------------------------------------------------
# Minimax solver
# ---------------------------------------------------------------------------

def solve_minimax(N, w_by_k, lower=0.0, upper=1.0, skewed_only=True,
                  odd=True, enforce_box=True, reg=0.0, solver=None,
                  verbose=False, verify_all=False):

    if N < 2:
        raise ValueError("need N >= 2")
    if upper <= lower:
        raise ValueError("need lower < upper")

    rng = upper - lower
    w_by_k = normalize_weights(N, w_by_k)
    sizes = get_sizes(N, w_by_k)

    if odd:
        check_symmetry(N, w_by_k)

    sigs = skewed_signatures(N) if skewed_only else all_signatures(N)

    installed = cp.installed_solvers()
    if solver is None:
        if "ECOS" in installed:
            solver = "ECOS"
        elif "SCS" in installed:
            solver = "SCS"
        else:
            raise RuntimeError("no supported solver found")

    grid, grid_index, theta_vec, P = build_matrices(N, sizes, sigs)
    G = len(grid)

    d = cp.Variable(G)
    t = cp.Variable(nonneg=True)

    constraints = []

    if enforce_box:
        constraints += [d >= -1, d <= 1]

    if odd:
        seen = set()
        for j, (nt, st, sc) in enumerate(grid):
            if j in seen:
                continue
            nc = N - nt
            swap = (nc, sc, st)
            jswap = grid_index[swap]
            if jswap == j:
                constraints.append(d[j] == 0)
            else:
                constraints.append(d[j] + d[jswap] == 0)
            seen.add(j)
            seen.add(jswap)

    theta_sq = theta_vec ** 2
    risk_vec = (theta_sq
                + P @ cp.square(d)
                - 2 * cp.multiply(theta_vec, P @ d))
    risk_constraint = (risk_vec <= t)
    constraints.append(risk_constraint)

    prob = cp.Problem(cp.Minimize(t + reg * cp.sum_squares(d)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if d.value is None or t.value is None:
        raise RuntimeError(f"solver failed: {prob.status}")

    d_vals = np.array(d.value).reshape(-1)
    worst_risk_norm = float(t.value)

    lam = np.asarray(risk_constraint.dual_value).reshape(-1)
    lam = np.maximum(lam, 0.0)
    if lam.sum() > 0:
        lam /= lam.sum()

    lfp = [(sigs[i], float(lam[i]))
           for i in range(len(lam)) if lam[i] > 1e-5]

    delta_orig = d_vals * rng
    worst_risk_orig = worst_risk_norm * (rng ** 2)

    sol = {grid[j]: float(delta_orig[j]) for j in range(G)}

    out = {
        "N": N,
        "w_by_k": w_by_k,
        "odd": odd,
        "skewed_only": skewed_only,
        "grid": grid,
        "grid_index": grid_index,
        "sizes": sizes,
        "delta_norm": d_vals,
        "worst_risk_norm": worst_risk_norm,
        "delta_orig": delta_orig,
        "worst_risk_orig": worst_risk_orig,
        "sol": sol,
        "lfp": lfp,
        "status": prob.status,
        "solver": solver,
    }

    if verify_all:
        all_sigs = all_signatures(N)
        _, _, theta_all, P_all = build_matrices(N, sizes, all_sigs)
        risks_all = (theta_all ** 2
                     + P_all @ (d_vals ** 2)
                     - 2 * theta_all * (P_all @ d_vals))
        out["verified_worst_risk_norm"] = float(np.max(risks_all))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Minimax estimator of the ATE under bounded outcomes")

    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--upper", type=float, default=1.0)

    ap.add_argument("--m", type=int, default=None,
                    help="uniform design: treat exactly m units")
    ap.add_argument("--k", type=int, default=None,
                    help="MOU design: treat N/2+k or N/2-k with prob 0.5 each")

    ap.add_argument("--all_states", action="store_true",
                    help="use all signatures (not just skewed)")
    ap.add_argument("--no_equivariance", action="store_true",
                    help="disable equivariance (sign-flip) constraint")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--verify_all", action="store_true",
                    help="verify worst-case risk over all signatures")
    ap.add_argument("--solver", type=str, default=None,
                    help="CVXPY solver name (e.g. ECOS, SCS)")

    args = ap.parse_args()

    if args.m is not None and args.k is not None:
        raise SystemExit("specify --m or --k, not both")

    if args.m is not None:
        w_by_k = uniform_weights(args.N, args.m)
    elif args.k is not None:
        w_by_k = mou_weights(args.N, args.k)
    else:
        w_by_k = balanced_weights(args.N)

    out = solve_minimax(
        N=args.N,
        w_by_k=w_by_k,
        lower=args.lower,
        upper=args.upper,
        skewed_only=(not args.all_states),
        odd=(not args.no_equivariance),
        verbose=args.verbose,
        verify_all=args.verify_all,
        solver=args.solver,
    )

    print(f"N={out['N']}  bounds=[{args.lower}, {args.upper}]")
    print(f"solver={out['solver']}  status={out['status']}")
    print()

    sol = out["sol"]
    sizes = out["sizes"]

    for nt, nc, w in sizes:
        print(f"--- nt={nt}, nc={nc}, weight={w:.3g} ---")
        header = "  st\\sc " + " ".join(f"{sc:>8d}" for sc in range(nc + 1))
        print(header)
        for st in range(nt + 1):
            row = f"  {st:4d}  "
            row += " ".join(f"{sol[(nt, st, sc)]:8.4f}"
                            for sc in range(nc + 1))
            print(row)
        print()

    if out["odd"]:
        print("(equivariant: delta(nt,st,sc) = -delta(nc,sc,st))")
        print()

    print(f"worst-case risk: {out['worst_risk_orig']:.6g}")

    if "verified_worst_risk_norm" in out:
        rng = args.upper - args.lower
        verified_orig = out["verified_worst_risk_norm"] * rng ** 2
        print(f"verified worst-case risk (all states): {verified_orig:.6g}")

    if out["lfp"]:
        print(f"\nleast-favorable prior ({len(out['lfp'])} signatures):")
        for sig, w in out["lfp"]:
            theta = signature_theta(sig, out["N"])
            print(f"  {sig}  theta={theta:.4f}  weight={w:.6g}")


if __name__ == "__main__":
    main()
