"""Conditional terminal-law calculations, independent of acquisition and plotting."""
import math
import numpy as np
from scipy.optimize import linprog


def normalise(quotes, spot, discount):
    if not (math.isfinite(spot) and spot > 0 and math.isfinite(discount) and discount > 0):
        raise ValueError("Positive finite spot and discount required")
    return [{**q, "k": discount*q["strike"]/spot,
             "lo": q["bid"]/spot if q.get("bid") is not None else None,
             "hi": q["ask"]/spot if q.get("ask") is not None else None} for q in quotes]


def bounds(quotes):
    puts = [(1-q["k"]+q["hi"], q["instrument"]) for q in quotes if q["type"] == "put" and q["hi"] is not None]
    calls = [(q["hi"], q["instrument"]) for q in quotes if q["type"] == "call" and q["hi"] is not None]
    up, ip = min(puts) if puts else (None, None)
    uc, ic = min(calls) if calls else (None, None)
    return {"U_put_raw": up, "U_put": min(1., up) if up is not None else None,
            "put_instrument": ip, "U_call_raw": uc, "call_instrument": ic,
            "U_combined": min([1.] + [u for u in [up, uc] if u is not None]),
            "negative_put_inconsistency": up is not None and up < -1e-8}


def parity(quotes, tolerance=1e-8):
    pairs = {}
    for q in quotes:
        pairs.setdefault(q["k"], {})[q["type"]] = q
    out = []
    for k, pair in sorted(pairs.items()):
        if "call" not in pair or "put" not in pair:
            continue
        c, p = pair["call"], pair["put"]
        low = c["lo"]-p["hi"] if c["lo"] is not None and p["hi"] is not None else None
        high = c["hi"]-p["lo"] if c["hi"] is not None and p["lo"] is not None else None
        theoretical = 1-k
        gap = max([0.] + ([low-theoretical] if low is not None else [])
                  + ([theoretical-high] if high is not None else []))
        out.append({"k": k, "parity_low": low, "parity_high": high,
                    "theoretical": theoretical, "violation": gap, "compatible": gap <= tolerance})
    return out


def verify(support, probabilities, quotes, defect, mode="AB", tolerance=1e-8):
    """Scalar payoff repricing, independent of the LP constraint matrices."""
    support, probabilities = list(map(float, support)), list(map(float, probabilities))
    if (not support or len(support) != len(probabilities)
            or not all(math.isfinite(v) for v in support+probabilities+[defect])
            or not 0 <= defect <= 1):
        raise ValueError("Finite, aligned, nonempty support and mass arrays and defect in [0,1] required")
    total = math.fsum(probabilities)
    mean = math.fsum(x*p for x, p in zip(support, probabilities))
    repriced = []
    for q in quotes:
        if mode == "A" and q["type"] != "put":
            continue
        fundamental = math.fsum(p*max(q["k"]-x if q["type"] == "put" else x-q["k"], 0.)
                                for x, p in zip(support, probabilities))
        price = fundamental + (1-mean if q["type"] == "call" else 0.)
        low_res = price-q["lo"] if q["lo"] is not None else None
        high_res = q["hi"]-price if q["hi"] is not None else None
        violation = max([0.] + ([-low_res] if low_res is not None else [])
                        + ([-high_res] if high_res is not None else []))
        repriced.append({"instrument": q["instrument"], "type": q["type"], "k": q["k"],
                         "bid_normalised": q["lo"], "ask_normalised": q["hi"],
                         "fundamental": fundamental, "price": price,
                         "bid_slack": low_res, "ask_slack": high_res, "violation": violation})
    error = max([abs(total-1), abs(mean-(1-defect)), max(0., -min(probabilities)),
                 max(0., -min(support)), max(0., mean-1)] + [r["violation"] for r in repriced])
    return {"valid": error <= tolerance, "probability_sum": total, "mean": mean,
            "defect": 1-mean, "max_error": error, "tolerance": tolerance, "repriced": repriced}


def witness(quotes, defect, mode="AB", ceilings=(2, 5, 20, 100, 1000), tolerance=1e-8):
    active = [q for q in quotes if mode == "AB" or q["type"] == "put"]
    if not active:
        return {"status": "unresolved_no_quotes", "attempts": []}
    attempts = []
    for ceiling in ceilings:
        # All payoff knots plus an expanding tail; no claim of unbounded infeasibility.
        knots = sorted({0., 1-defect, 1.} | {q["k"] for q in active})
        top = max(float(ceiling), max(knots)*1.01)
        x = np.unique(np.r_[knots, np.linspace(0, max(knots), 101), top])
        upper, rhs = [], []
        for q in active:
            row = np.maximum(q["k"]-x, 0) if q["type"] == "put" else np.maximum(x-q["k"], 0)
            offset = defect if q["type"] == "call" else 0.
            if q["hi"] is not None:
                upper.append(row); rhs.append(q["hi"]-offset)
            if q["lo"] is not None:
                upper.append(-row); rhs.append(-(q["lo"]-offset))
        result = linprog(np.zeros(len(x)), A_ub=np.array(upper) if upper else None,
                         b_ub=np.array(rhs) if upper else None,
                         A_eq=np.array([np.ones(len(x)), x]), b_eq=[1., 1-defect],
                         bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9,
                                  "dual_feasibility_tolerance": 1e-9})
        attempt = {"ceiling": top, "status": int(result.status), "message": result.message}
        attempts.append(attempt)
        if result.success:
            # Preserve every nonzero mass; do not round small tail probabilities away.
            mask = result.x != 0
            xs, ps = x[mask], result.x[mask]
            checked = verify(xs, ps, active, defect, mode, tolerance)
            attempt["independent_max_error"] = checked["max_error"]
            if checked["valid"]:
                return {"status": "admitted_by_checked_witness", "target_defect": defect,
                        "mode": mode, "support": xs.tolist(), "probabilities": ps.tolist(),
                        "validation": checked, "attempts": attempts}
    return {"status": "unresolved_on_searched_supports", "target_defect": defect,
            "mode": mode, "attempts": attempts}
