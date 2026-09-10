"""Read-only public production acquisition; no credentials or trading methods."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Client:
    def __init__(self, folder, config):
        self.folder, self.config = folder, config
        self.lock = threading.Lock()
        self.last = 0.0

    def get(self, label, method, **params):
        with self.lock:
            delay = self.config["request_spacing_seconds"] - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
        url = self.config["api_base"] + "/public/" + method + "?" + urllib.parse.urlencode(params)
        started = utc()
        meta = {"endpoint": method, "params": params, "url": url, "retrieval_started": started}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Bath-option-reliability-pilot/1.0"})
            with urllib.request.urlopen(req, timeout=self.config["request_timeout_seconds"]) as response:
                raw = response.read()
                meta.update(http_status=response.status, response_headers=dict(response.headers))
            payload = json.loads(raw)
            meta["sha256"] = hashlib.sha256(raw).hexdigest()
            raw_path = self.folder / (label + ".response.json")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            if payload.get("testnet") is not False:
                raise RuntimeError("Production testnet=false flag missing or false environment assertion")
            result = payload["result"]
        except Exception as exc:
            meta["error"] = str(exc)
            result = None
        meta["retrieval_finished"] = utc()
        save(self.folder / (label + ".meta.json"), meta)
        return {"result": result, "meta": meta}


def index(client, label):
    return client.get(label, "get_index_price", index_name="btc_usdc")


def choose_expiries(expiries, now, config):
    """Prospective horizon bands allow unavailable targets to remain missing."""
    used, selected, missing = set(), [], []
    for target in config["target_days"]:
        lower, upper = config["horizon_bands_days"][str(target)]
        candidates = [e for e in expiries if e not in used and lower <= (e-now)/86400000 <= upper]
        if not candidates:
            missing.append({"target_days": target, "band_days": [lower, upper],
                            "reason": "no eligible expiry within predeclared horizon band"})
            continue
        expiry = min(candidates, key=lambda e: (abs((e-now)/86400000-target), e))
        used.add(expiry)
        selected.append((target, expiry))
    return sorted(selected, key=lambda pair: pair[1]), missing


def batch(client, name, instruments, config):
    before = index(client, name + "/reference_before")
    started = utc()
    tick = time.monotonic()
    def fetch(spec):
        record = client.get(name + "/" + spec["instrument_name"], "get_order_book",
                            instrument_name=spec["instrument_name"], depth=1)
        return {"spec": spec, **record}
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        books = list(pool.map(fetch, instruments))
    after = index(client, name + "/reference_after")
    return {"name": name, "started": started, "finished": utc(),
            "elapsed_seconds": time.monotonic() - tick,
            "reference_before": before, "reference_after": after, "books": books}


def initial(client, config):
    cat = client.get("inventory/instruments", "get_instruments", currency="USDC", kind="option", expired="false")
    summaries = client.get("inventory/summaries", "get_book_summary_by_currency", currency="USDC", kind="option")
    futures = client.get("inventory/futures", "get_book_summary_by_currency", currency="USDC", kind="future")
    if cat["result"] is None or summaries["result"] is None:
        return {"status": "stopped_data_access", "inventory": cat, "summaries": summaries, "batches": []}
    now = time.time() * 1000
    specs = [s for s in cat["result"] if s.get("base_currency") == "BTC"
             and s.get("settlement_currency") == "USDC" and s.get("quote_currency") == "USDC"
             and s.get("instrument_type") == "linear" and s.get("is_active")
             and s.get("expiration_timestamp", 0) > now]
    summary = {s["instrument_name"]: s for s in summaries["result"]}
    expiries = sorted({s["expiration_timestamp"] for s in specs})
    selected = []
    choices, unavailable = choose_expiries(expiries, now, config)
    for target, expiry in choices:
        chain = [s for s in specs if s["expiration_timestamp"] == expiry]
        def screen(s):
            q = summary.get(s["instrument_name"], {})
            return ((q.get("bid_price") is not None and q["bid_price"] >= 0)
                    or (q.get("ask_price") is not None and q["ask_price"] > 0))
        strikes = sorted({s["strike"] for s in chain if screen(s)})
        n = min(len(strikes), config["max_distinct_strikes_per_expiry"])
        ranks = sorted({round(i*(len(strikes)-1)/(n-1)) for i in range(n)}) if n > 1 else list(range(n))
        chosen = {strikes[i] for i in ranks}
        selected.append({"target_days": target, "expiry": expiry,
                         "actual_days_at_inventory": (expiry-now)/86400000,
                         "listed_contracts": len(chain), "screened_strikes": len(strikes),
                         "selected_strikes": sorted(chosen),
                         "instruments": [s for s in chain if s["strike"] in chosen]})
    selection = {"frozen_at": utc(), "rule": config["selection_rule"], "expiries": selected,
                 "unavailable_targets": unavailable,
                 "all_eligible_specs": specs, "summary": summary, "futures": futures}
    save(client.folder / "selection.json", selection)
    batches = []
    for sel in selected:
        name = "expiry_" + str(sel["expiry"])
        b = batch(client, name, sel["instruments"], config)
        b["target_days"] = sel["target_days"]
        b["expiry"] = sel["expiry"]
        batches.append(b)
        print(name, len(b["books"]), "books", round(b["elapsed_seconds"], 2), "seconds", flush=True)
    return {"status": "collected" if batches else "stopped_no_linear_expiries",
            "selection": selection, "batches": batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recheck", type=Path, help="JSON list of influential instrument specifications")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "raw" / (("recheck_" if args.recheck else "initial_") + stamp)
    folder.mkdir(parents=True)
    save(folder / "config.json", config)
    client = Client(folder, config)
    if args.recheck:
        specs = json.loads(args.recheck.read_text())
        batches = []
        for expiry in sorted({s["expiration_timestamp"] for s in specs}):
            b = batch(client, "expiry_"+str(expiry), [s for s in specs if s["expiration_timestamp"] == expiry], config)
            b["expiry"] = expiry
            batches.append(b)
        result = {"status": "recheck_collected", "batches": batches, "source": str(args.recheck)}
    else:
        result = initial(client, config)
    save(folder / "session.json", {"created": utc(), "config": config, **result})
    print(folder, flush=True)


if __name__ == "__main__":
    main()
