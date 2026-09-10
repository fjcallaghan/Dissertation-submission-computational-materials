"""Offline replay, quote-quality gates, witnesses and compact pilot exhibits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core import bounds, normalise, parity, verify, witness


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def audit_raw(folder):
    checked = []
    for meta_path in sorted(folder.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if "sha256" not in meta:
            continue
        response = meta_path.with_name(meta_path.name.replace(".meta.json", ".response.json"))
        digest = hashlib.sha256(response.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError("Raw response checksum mismatch: " + str(response))
        checked.append({"file": str(response.resolve().relative_to(ROOT)), "sha256": digest})
    return checked


def audit_session(folder, session):
    """Tie convenient session inputs back to the preserved response bytes."""
    if json.loads((folder / "config.json").read_text()) != session["config"]:
        raise ValueError("Session/config mismatch")
    by_hash = {}
    for path in folder.rglob("*.response.json"):
        raw = path.read_bytes()
        by_hash[hashlib.sha256(raw).hexdigest()] = json.loads(raw)
    def visit(value):
        if isinstance(value, dict):
            if "result" in value and "meta" in value and value["result"] is not None:
                original = by_hash[value["meta"]["sha256"]]
                if original.get("testnet") is not False or original["result"] != value["result"]:
                    raise ValueError("Session record differs from production raw response")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(session)
    if "selection" in session:
        if json.loads((folder / "selection.json").read_text()) != session["selection"]:
            raise ValueError("Frozen selection mismatch")
        catalogue = json.loads((folder / "inventory/instruments.response.json").read_text())["result"]
        specs = {s["instrument_name"]: s for s in catalogue}
        for batch in session["batches"]:
            for book in batch["books"]:
                if book["spec"] != specs[book["spec"]["instrument_name"]]:
                    raise ValueError("Instrument specification differs from raw catalogue")


def discount_diagnostic(quotes, spot, years, shift=0.):
    """Necessary parity D interval only; never used to choose a pricing input."""
    by_strike = {}
    for q in quotes:
        by_strike.setdefault(q["strike"], {})[q["type"]] = q
    low, high = 0., None
    for strike, pair in by_strike.items():
        if "put" not in pair or "call" not in pair:
            continue
        c, p = pair["call"], pair["put"]
        if c["ask"] is not None and p["bid"] is not None:
            low = max(low, (spot*(1+shift)-c["ask"]+p["bid"])/strike)
        if c["bid"] is not None and p["ask"] is not None:
            bound = (spot*(1+shift)-c["bid"]+p["ask"])/strike
            high = bound if high is None else min(high, bound)
    compatible = high is None or high > 0 and low <= high
    return {"reference_shift": shift, "D_lower": low, "D_upper": high,
            "overlap": compatible,
            "rate_lower": -math.log(high)/years if compatible and high is not None else None,
            "rate_upper": -math.log(low)/years if compatible and low > 0 else None,
            "interpretation": "Necessary parity diagnostic only; not an estimated discount rate or a feasibility certificate."}


def clean_batch(batch, config):
    refs = [r.get("result", {}).get("index_price") if r.get("result") else None
            for r in [batch["reference_before"], batch["reference_after"]]]
    issues = []
    if any(not finite(r) or r <= 0 for r in refs):
        return [], [{"scope": "batch", "reason": "missing positive reference"}], {"usable": False}
    spot = sum(refs)/2
    all_refs = refs + [b["result"]["index_price"] for b in batch["books"]
                       if b.get("result") and finite(b["result"].get("index_price"))]
    move = (max(all_refs)-min(all_refs))/spot
    if batch["elapsed_seconds"] > config["max_batch_seconds"]:
        issues.append({"scope": "batch", "reason": "collection span exceeds 30-second rule"})
    if move > config["max_reference_movement_fraction"]:
        issues.append({"scope": "batch", "reason": "reference movement exceeds predeclared 0.1%"})
    batch_ok = not issues
    rows, ages = [], []
    for record in batch["books"]:
        spec, book, meta = record["spec"], record["result"], record["meta"]
        name = spec["instrument_name"]
        reasons = []
        if not book:
            issues.append({"instrument": name, "reason": "request failure: " + meta.get("error", "unknown")})
            continue
        retrieved = timestamp(meta["retrieval_finished"])
        if not finite(book.get("timestamp")):
            reasons.append("missing book timestamp")
            age = None
        else:
            age = retrieved-book["timestamp"]/1000
            ages.append(age)
            if age > config["max_book_age_seconds"] or age < -2:
                reasons.append("stale book or clock inconsistency")
        if spec["expiration_timestamp"]/1000 <= retrieved:
            reasons.append("expired")
        if book.get("state") != "open":
            reasons.append("book not open")
        if not (spec.get("contract_size") == 1 and spec.get("price_index") == "btc_usdc"
                and spec.get("instrument_type") == "linear" and spec.get("settlement_currency") == "USDC"
                and spec.get("quote_currency") == "USDC" and spec.get("base_currency") == "BTC"
                and spec.get("option_type") in ["put", "call"] and finite(spec.get("strike")) and spec["strike"] > 0):
            reasons.append("unsupported contract mapping")
        sides = {}
        for side, key in [("bid", "bids"), ("ask", "asks")]:
            level = book.get(key) or []
            value, amount = None, None
            if level:
                if (len(level[0]) >= 2 and finite(level[0][0]) and finite(level[0][1])
                        and level[0][0] >= 0 and level[0][1] > 0):
                    value, amount = level[0][:2]
                    if side == "ask" and value == 0:
                        reasons.append("zero ask requires investigation")
                else:
                    issues.append({"instrument": name, "side": side, "reason": "malformed price or nonpositive displayed size; side excluded"})
            else:
                issues.append({"instrument": name, "side": side, "reason": "missing quoted side"})
            sides[side] = value
            sides[side+"_amount"] = amount
        if sides["bid"] is not None and sides["ask"] is not None and sides["bid"] > sides["ask"]:
            reasons.append("crossed book")
        if sides["bid"] is None and sides["ask"] is None:
            reasons.append("no usable quoted sides")
        if reasons:
            issues.extend({"instrument": name, "reason": r} for r in reasons)
            continue
        spread = sides["ask"]-sides["bid"] if sides["bid"] is not None and sides["ask"] is not None else None
        rows.append({"instrument": name, "type": spec["option_type"], "expiry": spec["expiration_timestamp"],
                     "strike": spec["strike"], "contract_size": spec["contract_size"],
                     "quote_currency": spec["quote_currency"], "settlement_currency": spec["settlement_currency"],
                     **sides, "spread": spread, "constraint": "two-sided" if spread is not None else "one-sided",
                     "book_timestamp": book["timestamp"], "retrieved": meta["retrieval_finished"], "age_seconds": age,
                     "index_price": book.get("index_price"), "underlying_index": book.get("underlying_index"),
                     "underlying_price_metadata": book.get("underlying_price"), "open_interest": book.get("open_interest"),
                     "volume": book.get("stats", {}).get("volume"), "mark_price_metadata": book.get("mark_price"),
                     "exchange_interest_metadata": book.get("interest_rate"), "greeks_metadata": json.dumps(book.get("greeks")),
                     "tight_sample": spread is not None and spread/spot <= config["tight_spread_fraction_of_spot"]})
    return (rows if batch_ok else []), issues, {"usable": batch_ok, "spot": spot,
                "reference_before": refs[0], "reference_after": refs[1], "reference_min": min(all_refs),
                "reference_max": max(all_refs), "reference_movement_fraction": move,
                "elapsed_seconds": batch["elapsed_seconds"], "max_book_age_seconds": max(ages) if ages else None}


def analyse_batch(batch, config, out):
    quotes, exclusions, quality = clean_batch(batch, config)
    label = datetime.fromtimestamp(batch["expiry"]/1000, timezone.utc).strftime("%Y-%m-%d")
    folder = out / label
    table(folder / "quotes.csv", quotes)
    table(folder / "exclusions.csv", exclusions)
    result = {"expiry": label, "target_days": batch.get("target_days"), "quality": quality,
              "quote_count": len(quotes), "exclusion_records": len(exclusions), "witnesses": {}}
    if not quotes:
        result["status"] = "stopped_quality_or_no_quotes"
        return result, []
    spot = quality["spot"]
    years = (batch["expiry"]/1000 - timestamp(batch["started"]))/86400/365
    strikes = sorted({q["strike"] for q in quotes})
    result.update(days=years*365, distinct_strikes=len(strikes), strike_min=min(strikes), strike_max=max(strikes),
                  below_reference=sum(k < spot for k in strikes), above_reference=sum(k > spot for k in strikes),
                  status="conditional_calculations", assumptions=config["conventions"])
    qn = normalise(quotes, spot, math.exp(-config["baseline_rate"]*years))
    result["bounds"] = bounds(qn)
    parity_rows = parity(qn)
    table(folder / "parity.csv", parity_rows)
    result["parity"] = {"pairs": len(parity_rows), "violations": sum(not p["compatible"] for p in parity_rows),
                        "max_violation_fraction_of_spot": max([p["violation"] for p in parity_rows], default=0.)}
    result["discount_diagnostics"] = [discount_diagnostic(quotes, spot, years, shift)
        for shift in [-config["reference_sensitivity_fraction"], 0., config["reference_sensitivity_fraction"]]]
    table(folder / "parity_discount_diagnostic.csv", result["discount_diagnostics"])
    samples = {"all_selected": quotes,
               "remove_highest_25pct_strikes": [q for q in quotes if q["strike"] in strikes[:len(strikes)-math.ceil(.25*len(strikes))]],
               "tighter_spreads": [q for q in quotes if q["tight_sample"]]}
    sensitivities = []
    for sample, qs in samples.items():
        for rate in config["annual_continuous_rates"]:
            for shift in [-config["reference_sensitivity_fraction"], 0., config["reference_sensitivity_fraction"]]:
                norm = normalise(qs, spot*(1+shift), math.exp(-rate*years))
                p = parity(norm)
                sensitivities.append({"sample": sample, "rate": rate, "reference_shift": shift,
                                      "quotes": len(qs), "parity_violations": sum(not r["compatible"] for r in p), **bounds(norm)})
    result["sensitivities"] = sensitivities
    table(folder / "sensitivity.csv", sensitivities)
    for mode in ["A", "AB"]:
        upper = result["bounds"]["U_put"] if mode == "A" else result["bounds"]["U_combined"]
        incompatible = result["bounds"]["negative_put_inconsistency"] or (mode == "AB" and result["parity"]["violations"] > 0)
        ws = []
        for defect in [0.] + config["reference_defects"]:
            if incompatible:
                w = {"status": "inconsistent_pricing_assumptions", "target_defect": defect, "mode": mode}
            elif upper is not None and defect > upper+config["feasibility_tolerance"]:
                w = {"status": "excluded_by_conditional_bound", "target_defect": defect, "mode": mode}
            else:
                w = witness(qn, defect, mode, config["support_ceilings"], config["feasibility_tolerance"])
            ws.append(w)
        if not incompatible and not any(w["status"] == "admitted_by_checked_witness" and w["target_defect"] > 0 for w in ws):
            # Constructive search, never a significance test or sharp-bound optimisation.
            if upper is not None and upper > 1e-6:
                for fraction in [.5, .1, .01]:
                    w = witness(qn, min(.005, upper*fraction), mode, config["support_ceilings"], config["feasibility_tolerance"])
                    w["selection"] = "constructive search below elementary bound"
                    ws.append(w)
                    if w["status"] == "admitted_by_checked_witness":
                        break
        for i, w in enumerate(ws):
            if w["status"] == "admitted_by_checked_witness":
                w["spot"] = spot
                w["discount"] = math.exp(-config["baseline_rate"]*years)
                w["tail_threshold"] = max(q["k"] for q in qn if mode == "AB" or q["type"] == "put")
                w["tail_probability"] = sum(p for x, p in zip(w["support"], w["probabilities"]) if x > w["tail_threshold"])
                w["tail_mean_contribution"] = sum(x*p for x, p in zip(w["support"], w["probabilities"]) if x > w["tail_threshold"])
                w["above_reference_probability"] = sum(p for x, p in zip(w["support"], w["probabilities"]) if x > 1.)
                w["above_reference_mean_contribution"] = sum(x*p for x, p in zip(w["support"], w["probabilities"]) if x > 1.)
                save(folder / f"witness_{mode}_{i}.json", w)
                table(folder / f"witness_{mode}_{i}_repricing.csv", w["validation"]["repriced"])
                table(folder / f"witness_{mode}_{i}_distribution.csv", [
                    {"X": x, "terminal_index_USDC": x*spot/w["discount"], "probability": p,
                     "mean_contribution": x*p} for x, p in zip(w["support"], w["probabilities"])])
                # Check monotonicity using saved witness, without solving again.
                loose = [{**q, "lo": None if q["lo"] is None else max(0., q["lo"]-.001),
                          "hi": None if q["hi"] is None else q["hi"]+.001} for q in qn]
                assert verify(w["support"], w["probabilities"], loose, w["target_defect"], mode)["valid"]
                assert verify(w["support"], w["probabilities"], qn[::2], w["target_defect"], mode)["valid"]
        result["witnesses"][mode] = ws
    names = {result["bounds"]["put_instrument"], result["bounds"]["call_instrument"]}
    result["influential_quotes"] = [q for q in quotes if q["instrument"] in names]
    specs = [r["spec"] for r in batch["books"] if r["spec"]["instrument_name"] in names]
    save(folder / "analysis.json", result)
    plot_pair(result, folder)
    return result, specs


def plot_pair(result, folder):
    for mode, ws in result["witnesses"].items():
        good = [w for w in ws if w["status"] == "admitted_by_checked_witness"]
        zero = next((w for w in good if w["target_defect"] == 0), None)
        positive = next((w for w in good if w["target_defect"] > 0), None)
        if zero is None or positive is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for w, color in zip([zero, positive], ["#156082", "#bd582c"]):
            x, p = np.array(w["support"]), np.array(w["probabilities"])
            label = f'd = {100*w["target_defect"]:.4g}%'
            axes[0].vlines(x, 0, p, color=color, alpha=.75)
            axes[0].scatter(x, p, color=color, s=28, label=label)
            order = np.argsort(x)
            sx = np.r_[0., x[order], max(max(x), w["tail_threshold"])*1.05]
            sy = np.r_[sum(x*p), np.cumsum((x*p)[order][::-1])[::-1], 0.]
            axes[1].step(sx, sy, where="pre", color=color, label=label)
        thresholds = sorted({0., zero["tail_threshold"]*1.05} | set(zero["support"]) | set(positive["support"]))
        differences = []
        for t in thresholds:
            tails = [math.fsum(x*p for x, p in zip(w["support"], w["probabilities"]) if x >= t)
                     for w in [zero, positive]]
            differences.append(100*(tails[1]-tails[0]))
        axes[2].step(thresholds, differences, where="pre", color="#bd582c", label="Positive-defect law minus zero-defect law")
        axes[2].axhline(0, color="#555555", linewidth=.7)
        for ax in axes:
            if max(thresholds) > 5:
                ax.set_xscale("symlog", linthresh=1.)
            ax.axvline(zero["tail_threshold"], color="#555555", linestyle="--", linewidth=1, label="Highest retained strike")
            ax.set_xlabel("Normalised terminal delivery index X")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Probability at each support point")
        axes[1].set_ylabel("E[X 1{X ≥ threshold}]")
        axes[2].set_ylabel("Difference in upper-tail mean (percentage points)")
        fig.suptitle(f'{result["expiry"]} · Assumption {mode} · Constructed compatible laws, not probability estimates', fontsize=11)
        fig.savefig(folder / f"distribution_pair_{mode}.png", dpi=180)
        plt.close(fig)


def controls(out):
    q = []
    for k, p, c in zip([.8, 1., 1.2], [.15, .25, .35], [.35, .25, .15]):
        for side, price in [("put", p), ("call", c)]:
            q.append({"instrument": f"toy-{k}-{side}", "type": side, "strike": k, "bid": price, "ask": price})
    q = normalise(q, 1, 1)
    exact = [verify([.5, top], [.5, .5], q, d) for d, top in [(0, 1.5), (.1, 1.3)]]
    fits = [witness(q, d) for d in [0, .1]]
    assert all(v["valid"] for v in exact)
    assert all(w["status"] == "admitted_by_checked_witness" for w in fits)
    equal = normalise([{"instrument": f"equal-{k}", "type": "call", "strike": k, "bid": .1, "ask": .1} for k in [1., 2.]], 1, 1)
    control = verify([.9], [1.], equal, .1)
    assert control["valid"]
    save(out / "exact_controls.json", {"toy_exact_distributions": exact, "toy_solver_witnesses": fits,
          "equal_positive_calls": control,
          "equal_call_analytic_argument": "C(1)-C(2)=E[min((X-1)+,1)]=0 forces X<=1 almost surely. Then C(1)=d=0.1, so d=0 is analytically excluded under AB. This concerns terminal laws only."})


def pct(value):
    return "—" if value is None else f"{100*value:.4f}%"


def report(results, session, out, rechecks):
    valid = sorted([r for r in results if r.get("bounds")], key=lambda r: r["expiry"])
    lines = ["# Option-data reliability pilot: decision memo", "",
             "Completed from public production quotes on " + session["created"][:10] + ".", "",
             "This is a conditional terminal-pricing study of the BTC-USDC delivery index. It does not estimate whether Bitcoin has a bubble. Thesis integration reports the baseline witnesses and their limitations separately from historical explosiveness.", "",
             "## Decision", "",
             "Use the put-only bounds and independently verified witnesses as a short supplementary empirical illustration, subject to the convention note. Do not replace Chapter 5. Treat an inconsistent full-call convention as a failed assumption check, not evidence of a defect.", "",
             "## Exhibit 1: quote quality and conventions", "",
             "The inventory rule selected distinct expiries in target order. Any large target-to-actual mismatch is a coverage limitation, not an observation at the target horizon. A nearest remaining expiry far short of 180 days must not be described as a six-month observation. Each expiry has its own coherent acquisition batch and reference value; there is no synchronised cross-maturity model.", "",
             "| Expiry | Target / actual days | Usable contracts / strikes | Strike range (USDC) | Batch seconds | Reference movement |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in valid:
        lines.append(f'| {r["expiry"]} | {r["target_days"]} / {r["days"]:.2f} | {r["quote_count"]} / {r["distinct_strikes"]} | {r["strike_min"]:,.0f}–{r["strike_max"]:,.0f} | {r["quality"]["elapsed_seconds"]:.2f} | {pct(r["quality"]["reference_movement_fraction"])} |')
    lines += ["", "| Bound-driving contract | Initial ask (USDC/BTC) | Ask size (BTC) | Spread (USDC/BTC) | Book age at retrieval (seconds) |",
              "|---|---:|---:|---:|---:|"]
    for r in valid:
        for q in r["influential_quotes"]:
            spread = "one-sided" if q["spread"] is None else f'{q["spread"]:g}'
            lines.append(f'| {q["instrument"]} | {q["ask"]:g} | {q["ask_amount"]:g} | {spread} | {q["age_seconds"]:.3f} |')
    lines += ["", "Prices are USDC per BTC, with multiplier one. S0 is the average of the before/after BTC-USDC index observations. D=1 is the baseline; the predeclared rate scenarios are −2%, 0%, 5% and 10% continuously compounded on ACT/365. Reference scenarios are ±0.1%. These are conditional inputs, not measured financing rates or statistical uncertainty intervals. See [CONVENTIONS.md](../CONVENTIONS.md).", "",
              "Quotes use displayed book levels with positive size. Missing sides remain missing; zero bids can supply lower constraints. The deterministic 25-strike budget samples the screened ladder by rank, including endpoints; unsampled strikes are logged and are not claimed to be fitted. Marks, greeks and forward inputs are metadata only. Full quote, size, timestamp, spread, exclusions and scenario tables are in each expiry directory.", "",
              "## Exhibit 2: conditional necessary upper restrictions", "",
              "| Expiry | A status | Put-only necessary U | AB status | Combined U if not contradicted | Parity violations |",
              "|---|---|---:|---|---:|---:|"]
    for r in valid:
        b = r["bounds"]
        a_bad = b["negative_put_inconsistency"]
        ab_bad = a_bad or r["parity"]["violations"] > 0
        a_witness = any(w["status"] == "admitted_by_checked_witness" for w in r["witnesses"].get("A", []))
        a_status = "inconsistent" if a_bad else "checked witness" if a_witness else "unresolved"
        lines.append(f'| {r["expiry"]} | {a_status} | {pct(b["U_put"])} | {"inconsistent" if ab_bad else "not contradicted by parity"} | {"not applicable" if ab_bad else pct(b["U_combined"])} | {r["parity"]["violations"]}/{r["parity"]["pairs"]} |')
    lines += ["", "A combined number in a parity-inconsistent row is an algebraic necessary restriction in an empty model, **not a valid identification interval**. Put-only bounds retain their separate assumptions. A negative put bound is an inconsistency and is never clipped to zero. Values below an upper bound are admitted only when supported by a saved and independently checked witness.", "",
              "| Expiry | Assumptions | d=0 | d=1% | d=5% | d=10% |", "|---|---|---|---|---|---|"]
    names = {"admitted_by_checked_witness": "admitted (witness)", "excluded_by_conditional_bound": "excluded (bound)",
             "inconsistent_pricing_assumptions": "inconsistent model", "unresolved_on_searched_supports": "unresolved"}
    for r in valid:
        for mode, ws in r["witnesses"].items():
            lines.append(f'| {r["expiry"]} | {mode} | ' + " | ".join(names.get(w["status"], w["status"]) for w in ws[:4]) + " |")
    lines += ["", "The parity diagnostic additionally intersects the discount-factor restrictions from every retained call/put pair. These quote-implied ranges diagnose a convention mismatch; they are not selected as economic discount inputs or used to force a mean constraint.", ""]
    for r in valid:
        diag = next(d for d in r["discount_diagnostics"] if d["reference_shift"] == 0)
        description = (f'necessary continuously compounded rate range {pct(diag["rate_lower"])} to {pct(diag["rate_upper"])} at the baseline reference' if diag["overlap"]
                       else 'no common positive discount factor passes pairwise parity at the baseline reference')
        lines.append(f'- {r["expiry"]}: {description}. This is not a full distribution-feasibility test.')
    lines += ["", "## Exhibit 3: checked terminal distributions", ""]
    for r in valid:
        for mode, ws in r["witnesses"].items():
            good = [w for w in ws if w["status"] == "admitted_by_checked_witness"]
            if good:
                descriptions = [f'd={pct(w["target_defect"])} (maximum normalised residual {w["validation"]["max_error"]:.2g}; tail probability {w["tail_probability"]:.6g}, tail mean contribution {w["tail_mean_contribution"]:.6g})' for w in good]
                lines.append(f'- {r["expiry"]}, {mode}: ' + "; ".join(descriptions) + ".")
                image = out / r["expiry"] / f"distribution_pair_{mode}.png"
                if image.exists():
                    lines += ["", f'![Compatible laws for {r["expiry"]} under {mode}]({r["expiry"]}/distribution_pair_{mode}.png)', ""]
    lines += ["", "The LP fixes total mass, nonnegativity and the target mean together. Calls, when included, receive the full defect correction. Every successful law is saved at full floating-point precision and repriced with a separate scalar calculation; tolerance is 1e−8 in normalised units. Support ceilings expand through 2, 5, 20, 100 and 1000, always beyond the largest strike. A failed grid search remains unresolved. No tail ceiling is used to assert a positive lower bound.", "",
              "The reported tail probabilities concern mass strictly beyond the largest retained strike. Where both laws have zero mass there, their means differ through changes inside the observed strike range that fit within quote spreads. The plots show the actual constructed laws and differences in upper-tail mean contributions, including when the distributions nearly overlap. JSON outputs also record mass and mean contribution above X=1.", "",
              "## Limited sensitivity and quote recheck", "",
              "| Expiry | Baseline put bound | Remove highest 25% of strikes | Tighter spreads | Put-bound range over rate/reference grid (all selected quotes) |",
              "|---|---:|---:|---:|---:|"]
    for r in valid:
        ss = r["sensitivities"]
        def base(sample):
            return next(x["U_put"] for x in ss if x["sample"] == sample and x["rate"] == 0 and x["reference_shift"] == 0)
        vals = [x["U_put"] for x in ss if x["sample"] == "all_selected" and x["U_put"] is not None]
        lines.append(f'| {r["expiry"]} | {pct(r["bounds"]["U_put"])} | {pct(base("remove_highest_25pct_strikes"))} | {pct(base("tighter_spreads"))} | {pct(min(vals))}–{pct(max(vals))} |')
    lines += ["", "Sensitivity numbers remain conditional necessary restrictions; feasibility is tested only at the baseline. Inconsistent scenarios must not be pooled into an identification interval. The tighter sample uses two-sided spreads at most 0.5% of spot; it is separately labelled and does not replace the main sample.", ""]
    if any(s["U_put"] is not None and s["U_put"] >= .01 for r in valid for s in r["sensitivities"] if s["sample"] == "all_selected"):
        lines += ["Exclusion of 1% does not survive every declared rate/reference scenario across these expiries. Strike-removal and tighter-spread rows show the separate effects of sample coverage. These are measured limits to reliability. A checked positive defect below 1% demonstrates ambiguity only at that reported scale; it does not admit the predeclared 1%, 5% or 10% thresholds.", ""]
    if rechecks:
        lines += ["A second observation session re-fetched the bound-driving contracts. The table compares the same contracts using fresh reference observations. Recheck bounds concern this influential subset, not a recollected full chain. See `recheck.csv` for quote changes and quality gates.", "",
                  "| Expiry | Recheck quality | Put restriction from rechecked driver | Call restriction from rechecked driver |",
                  "|---|---|---:|---:|"]
        for r in rechecks:
            lines.append(f'| {r["expiry"]} | {"pass" if r["quality"]["usable"] else "fail"} | {pct(r["bounds"]["U_put"])} | {pct(r["bounds"]["U_call_raw"])} |')
    else:
        lines += ["Influential-quote recollection has not yet been included. Run the documented recheck command before treating the pilot as complete."]
    lines += ["", "## Interpretation and limits", "",
              "What the live exercise adds is a measured restriction on the scale of a conditional mean defect and, where saved witnesses admit distinct values, a concrete ambiguity at observed strikes. It does not identify the true pricing measure or estimate tail probabilities. The two exact toy laws and the special equal-positive-call control pass separately. In the latter, C(1)−C(2)=E[min((X−1)+,1)]=0 forces X≤1 almost surely, and therefore d=C(1)=0.1: zero can sometimes be excluded analytically.", "",
              "Within-batch reference movement and individual book ages are measured, not assumed absent. Bounds near quote precision remain sensitive to synchronisation and the stated reference convention. Rechecks are quality checks, not independent samples or a basis for p-values. The observed index range does not bound every unobserved tick. Fees, averaging and index-to-traded-spot basis are deliberately explicit approximations, so these are not unconditional bounds on the theorem’s instantaneous Bitcoin spot process.", "",
              "An identification interval is not a confidence interval. These finite-support laws do not construct a continuous-time strict local martingale, establish cross-maturity consistency, test HK’s infinite-horizon property, or explain 2017/2021 episodes. The corrected Chapter 5 verification results remain the empirical baseline. No originality claim is made; [SOURCES.md](../../SOURCES.md) places this exercise alongside existing finite-strike moment-bound and bid–ask-consistency research.", "",
              "## Reproduction", "", "See [the pilot README](../README.md). Raw responses, request parameters, retrieval times, production flags and SHA-256 hashes are retained. Offline replay verifies the raw hashes before calculating the exhibits. The current run’s machine-readable outputs are in `summary.json`, and essential validation is in `exact_controls.json` and the test log."]
    memo = "\n".join(lines)+"\n"
    memo = memo.replace("](../CONVENTIONS.md)", "](../../CONVENTIONS.md)").replace("](../README.md)", "](../../README.md)")
    (out / "decision-memo.md").write_text(memo)
    if valid:
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        labels = [r["expiry"] for r in valid]
        values = [100*r["bounds"]["U_put"] if not r["bounds"]["negative_put_inconsistency"] else np.nan for r in valid]
        ax.bar(labels, values, color="#156082", width=.5)
        for i, r in enumerate(valid):
            if r["bounds"]["negative_put_inconsistency"]:
                ax.text(i, .2, "Inconsistent\nbaseline", ha="center", color="#a13d31")
            else:
                ax.text(i, values[i]+.15, pct(r["bounds"]["U_put"]), ha="center")
        for d in [1, 5, 10]:
            ax.axhline(d, color="#888888", linewidth=.8, linestyle="--")
            ax.text(2.5, d, f" {d}%", va="center", fontsize=9)
        ax.set_ylabel("Conditional upper bound (% of reference index)")
        ax.set_title("Put-only restriction under A · D = 1\nUpper bounds, not estimated defects or confidence intervals")
        ax.grid(axis="y", alpha=.15)
        fig.savefig(out / "conditional_bounds.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--recheck", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Separate replay destination to preserve delivered exhibits")
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    config = session["config"]
    out = args.output_dir or ROOT / "outputs" / args.session.parent.name
    out.mkdir(parents=True, exist_ok=True)
    manifest = audit_raw(args.session.parent)
    audit_session(args.session.parent, session)
    controls(out)
    results, influential = [], []
    for batch in session["batches"]:
        r, specs = analyse_batch(batch, config, out)
        results.append(r)
        influential.extend(specs)
    save(out / "influential_instruments.json", influential)
    table(out / "influential_quotes.csv", [{"reference": r["quality"]["spot"], **q}
          for r in results for q in r.get("influential_quotes", [])])
    # Inventory omissions stay distinct from rejected live quotes.
    omissions = []
    for sel in session.get("selection", {}).get("expiries", []):
        chosen = {i["instrument_name"] for i in sel["instruments"]}
        for spec in session["selection"]["all_eligible_specs"]:
            if spec["expiration_timestamp"] == sel["expiry"] and spec["instrument_name"] not in chosen:
                omissions.append({"instrument": spec["instrument_name"], "strike": spec["strike"],
                                  "reason": "not selected by frozen summary/rank budget rule"})
    table(out / "inventory_omissions.csv", omissions)
    rechecks = []
    if args.recheck:
        rc = json.loads(args.recheck.read_text())
        manifest += audit_raw(args.recheck.parent)
        audit_session(args.recheck.parent, rc)
        for b in rc["batches"]:
            qs, exclusions, quality = clean_batch(b, config)
            date = datetime.fromtimestamp(b["expiry"]/1000, timezone.utc).strftime("%Y-%m-%d")
            bs = bounds(normalise(qs, quality["spot"], 1.)) if qs else bounds([])
            rechecks.append({"expiry": date, "quality": quality, "bounds": bs, "quotes": qs, "exclusions": exclusions})
        initial_quotes = {q["instrument"]: q for r in results for q in r.get("influential_quotes", [])}
        table(out / "recheck.csv", [{"expiry": r["expiry"], "quality_pass": r["quality"]["usable"],
            "instrument": q["instrument"], "initial_ask": initial_quotes.get(q["instrument"], {}).get("ask"),
            "recheck_ask": q["ask"], "recheck_ask_amount": q["ask_amount"], "recheck_timestamp": q["book_timestamp"],
            "reference": r["quality"]["spot"], "spread": q["spread"], **r["bounds"]}
            for r in rechecks for q in r["quotes"]])
    save(out / "raw_manifest.json", manifest)
    save(out / "summary.json", {"source_session": str(args.session.resolve()),
          "recheck_session": str(args.recheck.resolve()) if args.recheck else None,
          "results": results, "rechecks": rechecks, "raw_responses_verified": len(manifest)})
    save(out / "run_metadata.json", {"analysed_at": datetime.now(timezone.utc).isoformat(),
          "session_sha256": hashlib.sha256(args.session.read_bytes()).hexdigest(),
          "recheck_sha256": hashlib.sha256(args.recheck.read_bytes()).hexdigest() if args.recheck else None,
          "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "scripts").glob("*.py")},
          "config_sha256": hashlib.sha256((args.session.parent / "config.json").read_bytes()).hexdigest()})
    report(results, session, out, rechecks)
    for r in results:
        print(r["expiry"], r.get("bounds"), r.get("parity"))
        for mode, ws in r["witnesses"].items():
            print(mode, [(w["target_defect"], w["status"]) for w in ws])
    print("Output:", out)


if __name__ == "__main__":
    main()
