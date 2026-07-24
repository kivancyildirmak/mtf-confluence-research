#!/usr/bin/env python3
"""
Qualification V1 — deterministic threshold sensitivity study harness.

WHAT THIS IS
------------
A lookahead-safe, fully deterministic engine that sweeps the market/bucket
evidence thresholds over a per-prediction event log and reports every metric
requested by the sensitivity study (PARTs 1, 3, 4). It reuses the exact decision
rule from qualify_ref.qualify (which mirrors Pine f_qualify line-for-line), so
the study and production classify identically.

WHAT THIS NEEDS
---------------
A per-prediction event log per symbol. Each prediction:
    dict(id, create_t, resolve_t, outcome, scB, tdB)
      create_t  : monotonic creation index/bar (chronological)
      resolve_t : resolution index/bar, or None if still active at chart end
      outcome   : "H" hit | "F" fail | "E" expired | "U" unresolved(active)
      scB       : score fine-bin 0..5   (f_scBin at creation)
      tdB       : target-distance bin 0..3 (f_tdBin at creation)
This log does NOT exist in the repo yet — it must be exported from Pine (see
the schema note at the bottom / the study report). Until then this file runs on
a clearly-labelled deterministic TOY dataset so the machinery is proven and the
output format is exact. TOY numbers are NOT real symbol behaviour.

WALK-FORWARD INTEGRITY (PART 2)
-------------------------------
Evidence is captured in ONE chronological replay pass. At each CREATE event a
prediction's evidence snapshot is taken from prior-RESOLVED counters only; the
prediction's own resolution happens strictly later on the timeline, so it can
never be in its own evidence. Threshold combinations are then evaluated against
these frozen snapshots — thresholds never touch the walk-forward, so lookahead
safety is structural and independent of the grid.
"""

import importlib.util
import os
import statistics

# ---- import the reference rule (same directory) ---------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("qualify_ref", os.path.join(_here, "qualify_ref.py"))
qr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qr)  # type: ignore

UNKNOWN, QUALIFIED, WEAK, REJECTED = 0, 1, 2, 3
REASON_NAMES = qr.REASON_NAMES

MARKET_GRID = [20, 30, 40, 50, 75, 100]
BUCKET_GRID = [10, 15, 20, 30, 40]

# score bins 0..5, dist bins 0..3 (matches f_scBin / f_tdBin ranges)
SC_BINS, TD_BINS = 6, 4
# joint level enumeration mirroring Pine's cbKindA=0 Score>= x cbKindB=2 Dist<=
JOINT_LVA = [1, 2, 3, 4, 5]   # Score>= thresholds
JOINT_LVB = [0, 1, 2]         # Dist<=  thresholds


# ===========================================================================
# PASS 1 — walk-forward evidence capture (threshold-independent, done once)
# ===========================================================================
def capture_snapshots(preds):
    """Replay the chronological timeline; freeze each prediction's as-of-creation
    evidence from prior-resolved counters only. Returns a list of snapshots, one
    per prediction, in creation order."""
    # merged event stream: (time, kind, pred)   kind 0=create sorts before 1=resolve
    events = []
    for p in preds:
        events.append((p["create_t"], 0, p))
        if p["resolve_t"] is not None and p["outcome"] in ("H", "F", "E"):
            events.append((p["resolve_t"], 1, p))
    # stable chronological order; at equal time, creations before resolutions so a
    # prediction created on the same bar an earlier one resolves still excludes it
    # only if it resolved strictly earlier — creations never see same-bar resolves.
    events.sort(key=lambda e: (e[0], e[1]))

    mH = mF = mE = 0
    sc_hit = [0] * SC_BINS; sc_fail = [0] * SC_BINS; sc_exp = [0] * SC_BINS
    td_hit = [0] * TD_BINS; td_fail = [0] * TD_BINS; td_exp = [0] * TD_BINS
    # resolved (scB, tdB, outcome) accumulation for joint combos
    j_hit = {}; j_fail = {}; j_exp = {}   # keyed (lvA, lvB)
    for a in JOINT_LVA:
        for b in JOINT_LVB:
            j_hit[(a, b)] = 0; j_fail[(a, b)] = 0; j_exp[(a, b)] = 0

    snaps = []
    for (_, kind, p) in events:
        if kind == 0:
            # CREATE — snapshot prior-resolved counters
            combos = []
            for a in JOINT_LVA:
                for b in JOINT_LVB:
                    combos.append((0, a, 2, b, j_hit[(a, b)], j_fail[(a, b)], j_exp[(a, b)]))
            snaps.append(dict(
                pid=p["id"], scB=p["scB"], tdB=p["tdB"], outcome=p["outcome"],
                market=(mH, mF, mE),
                combos=combos,
                anSc=dict(hit=sc_hit[:], fail=sc_fail[:], exp=sc_exp[:]),
                anTd=dict(hit=td_hit[:], fail=td_fail[:], exp=td_exp[:]),
            ))
        else:
            # RESOLVE — bump market, score bin, dist bin, and every joint combo
            # whose key the prediction satisfies (scB>=lvA and tdB<=lvB).
            o, s, d = p["outcome"], p["scB"], p["tdB"]
            if o == "H":
                mH += 1; sc_hit[s] += 1; td_hit[d] += 1
            elif o == "F":
                mF += 1; sc_fail[s] += 1; td_fail[d] += 1
            elif o == "E":
                mE += 1; sc_exp[s] += 1; td_exp[d] += 1
            for a in JOINT_LVA:
                for b in JOINT_LVB:
                    if s >= a and d <= b:
                        if o == "H": j_hit[(a, b)] += 1
                        elif o == "F": j_fail[(a, b)] += 1
                        elif o == "E": j_exp[(a, b)] += 1
    return snaps


# ===========================================================================
# PASS 2 — evaluate one (min_market, min_bucket) combo over frozen snapshots
# ===========================================================================
def _perturb_class(cond, bN, baseline, qual_lift, rej_lift):
    """Class under +1 hit and +1 fail added to the chosen bucket; also max |Δcond|."""
    h = round(cond * bN)
    def cls(lift):
        return QUALIFIED if lift >= qual_lift else REJECTED if lift <= rej_lift else WEAK
    cond_h = (h + 1) / (bN + 1)
    cond_f = h / (bN + 1)
    cls_h = cls(cond_h - baseline)
    cls_f = cls(cond_f - baseline)
    dmax = max(abs(cond_h - cond), abs(cond_f - cond))
    return cls_h, cls_f, dmax


def eval_combo(snaps, mm, mb):
    counts = {UNKNOWN: 0, QUALIFIED: 0, WEAK: 0, REJECTED: 0}
    reason_hist = {}
    joint_used = single_used = 0
    samples = []                 # resolved sample used (bN) for classified
    abs_lifts = []
    barely = exactly_min = 0     # PART 3
    flips = big_shift = 0
    # class hit/decisive tallies for QUALIFIED/WEAK/REJECTED hit rates
    cls_hit = {QUALIFIED: 0, WEAK: 0, REJECTED: 0}
    cls_dec = {QUALIFIED: 0, WEAK: 0, REJECTED: 0}
    base_hit = base_dec = 0

    for sn in snaps:
        r = qr.qualify(sn["scB"], sn["tdB"], sn["market"], sn["combos"], sn["anSc"], sn["anTd"],
                       min_market=mm, min_bucket=mb)
        st = r["status"]
        counts[st] += 1
        # market baseline population (prior-resolved decisive at creation)
        mHc, mFc, _mEc = sn["market"]
        base_hit += mHc; base_dec += (mHc + mFc)
        if st == UNKNOWN:
            reason_hist[r["reason"]] = reason_hist.get(r["reason"], 0) + 1
            continue
        # classified
        if r["bType"] == 1:
            joint_used += 1
        else:
            single_used += 1
        bN = r["bN"]; samples.append(bN); abs_lifts.append(abs(r["lift"]))
        if mb <= bN <= mb + 2:
            barely += 1
        if bN == mb:
            exactly_min += 1
        cls_h, cls_f, dmax = _perturb_class(r["cond"], bN, r["baseline"], 0.05, -0.05)
        if cls_h != st or cls_f != st:
            flips += 1
        if dmax > 0.05:
            big_shift += 1
        # did this prediction's own outcome hit?
        if sn["outcome"] in ("H", "F"):
            cls_dec[st] += 1
            if sn["outcome"] == "H":
                cls_hit[st] += 1

    total = len(snaps)
    classified = counts[QUALIFIED] + counts[WEAK] + counts[REJECTED]

    def rate(h, d):
        return (100.0 * h / d) if d > 0 else None

    return dict(
        mm=mm, mb=mb, total=total,
        Q=counts[QUALIFIED], W=counts[WEAK], U=counts[UNKNOWN], R=counts[REJECTED],
        coverage=(100.0 * classified / total) if total else 0.0,
        q_hit=rate(cls_hit[QUALIFIED], cls_dec[QUALIFIED]),
        w_hit=rate(cls_hit[WEAK], cls_dec[WEAK]),
        r_hit=rate(cls_hit[REJECTED], cls_dec[REJECTED]),
        baseline=rate(base_hit, base_dec),   # avg over creations of prior baseline
        avg_sample=(statistics.mean(samples) if samples else None),
        med_sample=(statistics.median(samples) if samples else None),
        pct_joint=(100.0 * joint_used / classified) if classified else None,
        pct_single=(100.0 * single_used / classified) if classified else None,
        reason_hist={REASON_NAMES[k]: v for k, v in sorted(reason_hist.items())},
        # PART 3 stability
        barely=barely, exactly_min=exactly_min,
        avg_abs_lift=(statistics.mean(abs_lifts) if abs_lifts else None),
        med_abs_lift=(statistics.median(abs_lifts) if abs_lifts else None),
        flip_rate=(100.0 * flips / classified) if classified else None,
        bigshift_rate=(100.0 * big_shift / classified) if classified else None,
    )


def run_symbol(name, preds):
    snaps = capture_snapshots(preds)
    rows = []
    for mm in MARKET_GRID:
        for mb in BUCKET_GRID:
            rows.append(eval_combo(snaps, mm, mb))
    return name, rows


# ===========================================================================
# Reporting
# ===========================================================================
def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def print_symbol_table(name, rows):
    print(f"\n===== {name} — full sensitivity grid (6 market x 5 bucket) =====")
    hdr = ("MKT BKT  TOT   Q   W   U   R   cov%  Qhit  Whit  Rhit  base  "
           "avgN medN  joint% flip%  >5pp%  UNKNOWN-reasons")
    print(hdr)
    for r in rows:
        rs = " ".join(f"{k}:{v}" for k, v in r["reason_hist"].items()) or "-"
        print(f"{r['mm']:>3} {r['mb']:>3} {r['total']:>4} "
              f"{r['Q']:>3} {r['W']:>3} {r['U']:>3} {r['R']:>3} "
              f"{_f(r['coverage']):>5} {_f(r['q_hit']):>5} {_f(r['w_hit']):>5} {_f(r['r_hit']):>5} "
              f"{_f(r['baseline']):>5} {_f(r['avg_sample'],1):>4} {_f(r['med_sample'],0):>4} "
              f"{_f(r['pct_joint']):>5} {_f(r['flip_rate']):>5} {_f(r['bigshift_rate']):>5}  {rs}")


def reliability_table():
    """PART 3 — data-INDEPENDENT reliability of a bucket decision at size n.
    Binomial SE at p=0.5 (worst case), and the shift one extra outcome causes.
    The decision band is +/-5pp; compare SE and per-outcome shift against it."""
    print("\n===== Reliability of a decision at bucket size n (data-independent) =====")
    print("   n   SE(p=.5)pp   1-outcome shift pp   note")
    for n in [10, 15, 20, 30, 40, 50, 75, 100]:
        se = 100.0 * (0.25 / n) ** 0.5
        shift = 100.0 * 0.5 / (n + 1)   # max |Δp| adding one outcome at p=.5
        note = "noise > band" if se > 5 else "SE within band"
        flip = " (1 sample can cross ±5pp band)" if shift > 5 else ""
        print(f"  {n:>3}   {se:>8.1f}     {shift:>12.1f}       {note}{flip}")


def feasibility_note(symbol_decisive):
    """PART 4 — REAL upper bound from marginals only. Walk-forward decisive-at-
    creation <= final decisive, so a symbol whose FINAL decisive < MIN_MARKET can
    NEVER classify anything (0% coverage) under that market threshold."""
    print("\n===== Market-gate feasibility from marginals (upper bound, REAL) =====")
    print("Symbol      finalDecisive   " + "  ".join(f"MM={m}" for m in MARKET_GRID))
    for sym, dec in symbol_decisive.items():
        cells = []
        for m in MARKET_GRID:
            cells.append(" possible" if dec >= m else "   0%    ")
        print(f"{sym:<12}{dec:>10}      " + "".join(cells))
    print("'possible' = gate CAN pass for the latest predictions; actual coverage")
    print("needs the walk-forward log. '0%' = structurally impossible (final < MM).")


# ===========================================================================
# Deterministic TOY dataset (NOT real symbol data — harness demonstration only)
# ===========================================================================
def toy_dataset(n=120, seed=1):
    """Fully deterministic (no RNG state leakage): construct a creation/resolution
    interleaving with ~3/4 decisive so low market thresholds classify and high
    ones do not — mirroring the shape of a mid-volume 1D symbol. Score/dist bins
    assigned by a fixed rotation. Outcomes fixed by index arithmetic."""
    preds = []
    for i in range(n):
        # decisive 3 of every 4; the 4th expires; a few tail ones stay active
        if i % 4 == 3:
            outcome = "E"
        else:
            outcome = "H" if (i * 7 + 3) % 5 < 3 else "F"   # ~60% of decisive hit
        resolve_t = None if i >= n - 6 else 2 * i + 40      # last 6 still active
        out = "U" if resolve_t is None else outcome
        preds.append(dict(id=i + 1, create_t=2 * i,
                          resolve_t=resolve_t, outcome=out,
                          scB=(i % SC_BINS), tdB=(i % TD_BINS)))
    return preds


def main():
    print(__doc__.strip().splitlines()[0])
    print("\n### TOY DEMONSTRATION — synthetic, NOT real symbol behaviour ###")
    name, rows = run_symbol("TOY(mid-vol)", toy_dataset())
    print_symbol_table(name, rows)
    reliability_table()
    # marginals the user reported (final decisive = hit + failed)
    feasibility_note({
        "AYGAZ": 87,    # 46 hit + 41 fail (CONFIRMED decisive)
        "XU100": 90,    # reported total; decisive <= 90 (upper bound)
        "GOZDE": 53,    # reported total; decisive <= 53 (upper bound)
        "Gold":  48,    # XAUTRYG-like reported total; decisive <= 48 (upper bound)
        "BTCUSD": 16,   # reported total; decisive <= 16 (upper bound)
    })
    print("\nNOTE: XU100/GOZDE/Gold/BTCUSD 'decisive' above are reported TOTALS used")
    print("as an upper bound; only AYGAZ's hit/fail split (87 decisive) is confirmed.")


if __name__ == "__main__":
    main()
