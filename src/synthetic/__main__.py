"""Synthetic-path runner — ``python -m src.synthetic``.

Generates GBM (null) and CEV (bubble) paths from ``config.yaml`` settings, prints
the martingale diagnostic (E[S_T] vs S_0), and saves the terminal values plus a
few sample paths to ``data/processed/`` for reuse.
"""

from __future__ import annotations

import sys

import pandas as pd

from ..config_loader import load_config
from .paths import (
    martingale_diagnostic,
    simulate_cev,
    simulate_evans_bubble,
    simulate_explosive_ar1,
    simulate_gbm,
    simulate_random_walk,
)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    s = cfg.synthetic
    common = dict(
        s0=float(s["s0"]), sigma=float(s["sigma"]),
        t_horizon=float(s["t_horizon"]), n_steps=int(s["n_steps"]),
        n_paths=int(s["n_paths"]), seed=int(s["seed"]),
    )

    gbm = simulate_gbm(**common)
    cev = simulate_cev(rho=float(s["cev_rho"]), **common)

    print(f"Synthetic paths — n_paths={common['n_paths']}, n_steps={common['n_steps']}, "
          f"sigma={common['sigma']}, seed={common['seed']}")
    for diag in (martingale_diagnostic(gbm), martingale_diagnostic(cev)):
        print(f"  [{diag['model']:>3}] E[S_T]={diag['E[S_T]_hat']:.4f} "
              f"(+/-{diag['std_error']:.4f})  ratio_to_S0={diag['ratio_to_s0']:.4f}  "
              f"absorbed@0={diag['frac_absorbed_zero']:.3f}")
    print("  Expect GBM ratio ~ 1.0 (true martingale); CEV ratio < 1.0 (bubble).")

    # Persist terminal values + a handful of sample paths.
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    term = pd.DataFrame({"gbm_terminal": gbm.terminal, "cev_terminal": cev.terminal})
    term.to_parquet(cfg.processed_dir / "synthetic_terminal.parquet", index=False)

    k = min(20, common["n_paths"])
    sample = pd.DataFrame({"t": gbm.t})
    for j in range(k):
        sample[f"gbm_{j}"] = gbm.S[j]
        sample[f"cev_{j}"] = cev.S[j]
    sample.to_parquet(cfg.processed_dir / "synthetic_sample_paths.parquet", index=False)

    print(f"\nSaved terminal values and {k} sample paths to {cfg.processed_dir}")

    # ----- Discrete-time controls for the GSADF explosive-root test -----
    # Ground truth for src.detect: the unit-root null must NOT be flagged, the
    # explosive and Evans series MUST be. One representative path each; a summary
    # of the terminal-vs-start growth makes the qualitative contrast visible.
    seed = int(s["seed"])
    n_disc = 400
    rw = simulate_random_walk(n_steps=n_disc, n_paths=1, seed=seed)
    ex = simulate_explosive_ar1(phi=1.05, n_steps=n_disc, n_paths=1, seed=seed)
    ev = simulate_evans_bubble(n_steps=n_disc, n_paths=1, seed=seed)
    print("\nDiscrete GSADF controls (1 path each, n_steps="
          f"{n_disc}):")
    for p in (rw, ex, ev):
        y = p.S[0]
        print(f"  [{p.model:>13}] start={y[0]:.2f} end={y[-1]:.2f} "
              f"max={y.max():.2f}  (validation target for src.detect)")
    disc = pd.DataFrame({
        "step": rw.t,
        "random_walk": rw.S[0],
        "explosive_ar1": ex.S[0],
        "evans_bubble": ev.S[0],
    })
    disc.to_parquet(cfg.processed_dir / "synthetic_discrete_controls.parquet", index=False)
    print(f"Saved discrete controls to {cfg.processed_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
