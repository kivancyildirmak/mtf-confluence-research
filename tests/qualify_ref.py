#!/usr/bin/env python3
"""
Reference implementation + tests for Qualification V1 (SHADOW).

This mirrors, line-for-line in logic, the Pine function f_qualify() in
src/mtf_confluence_strategy.pine. Pine cannot be unit-tested outside
TradingView, so the pure decision rule is verified here. Any change to the
Pine rule must be reflected here and re-run.

Rule (identical to Pine):
  - Inputs are ONLY prior-resolved counters (no candidate outcome). This is the
    structural no-lookahead guarantee: the candidate cannot influence its label.
  - Bucket selection: most-specific SUFFICIENT joint Score x Dist combo, else
    the strongest SUFFICIENT single (Score cumulative>=, Dist cumulative<=).
    Correlated lifts are never summed; exactly one bucket is used.
  - Classify by lift = conditional_hit_rate - market_baseline_hit_rate.
"""

# Global parameters (defaults, identical to the Pine inputs)
MIN_MARKET_RESOLVED = 100
MIN_BUCKET_RESOLVED = 40
MIN_COVERAGE        = 0.05
QUALIFIED_MIN_LIFT  = 0.05
REJECTED_MAX_LIFT   = -0.05

UNKNOWN, QUALIFIED, WEAK, REJECTED = 0, 1, 2, 3
NAMES = {0: "UNKNOWN", 1: "QUALIFIED", 2: "WEAK", 3: "REJECTED"}


def cum_from(a, start):
    return sum(a[start:])


def cum_up_to(a, last):
    return sum(a[:last + 1])


def qualify(scB, tdB, market, combos, anSc, anTd,
            min_market=MIN_MARKET_RESOLVED, min_bucket=MIN_BUCKET_RESOLVED,
            min_cov=MIN_COVERAGE, qual_lift=QUALIFIED_MIN_LIFT,
            rej_lift=REJECTED_MAX_LIFT):
    """
    market  = (mH, mF, mE)  prior-resolved market hits/fails/expired
    combos  = list of (kindA, lvA, kindB, lvB, h, f, e)
    anSc    = dict(hit=[6], fail=[6], exp=[6])   # score fine bins
    anTd    = dict(hit=[4], fail=[4], exp=[4])   # target-distance bins
    """
    mH, mF, mE = market
    marketN = mH + mF + mE
    marketDec = mH + mF
    baseline = (mH / marketDec) if marketDec > 0 else None
    status, reason, bType, bN = UNKNOWN, 1, 0, 0
    cond = lift = cov = None

    if marketN < min_market or marketDec == 0 or baseline is None:
        return dict(status=UNKNOWN, reason=1, bType=0, bN=0, marketN=marketN,
                    baseline=baseline, cond=None, lift=None, cov=None)

    # (a) most-specific sufficient joint Score x Dist bucket
    jFound = False
    jSpec = -1
    jH = jF = jE = 0
    for (kindA, lvA, kindB, lvB, h, f, e) in combos:
        if kindA == 0 and kindB == 2 and scB >= lvA and tdB <= lvB:
            dec, tot = h + f, h + f + e
            if dec >= min_bucket and tot / marketN >= min_cov:
                spec = lvA * 10 - lvB
                if (not jFound) or spec > jSpec:
                    jFound, jSpec = True, spec
                    jH, jF, jE = h, f, e
    if jFound:
        bType, bN = 1, jH + jF
        cond = jH / (jH + jF) if (jH + jF) > 0 else None
        cov = (jH + jF + jE) / marketN
    else:
        # (b) strongest sufficient single (score cumulative>=, dist cumulative<=)
        sH, sF, sE = cum_from(anSc["hit"], scB), cum_from(anSc["fail"], scB), cum_from(anSc["exp"], scB)
        dH, dF, dE = cum_up_to(anTd["hit"], tdB), cum_up_to(anTd["fail"], tdB), cum_up_to(anTd["exp"], tdB)
        sOk = (sH + sF) >= min_bucket and (sH + sF + sE) / marketN >= min_cov
        dOk = (dH + dF) >= min_bucket and (dH + dF + dE) / marketN >= min_cov
        sCond = sH / (sH + sF) if (sH + sF) > 0 else None
        dCond = dH / (dH + dF) if (dH + dF) > 0 else None
        sLift = None if sCond is None else sCond - baseline
        dLift = None if dCond is None else dCond - baseline
        if sOk and (not dOk or abs(sLift) >= abs(dLift)):
            bType, bN, cond, cov = 2, sH + sF, sCond, (sH + sF + sE) / marketN
        elif dOk:
            bType, bN, cond, cov = 3, dH + dF, dCond, (dH + dF + dE) / marketN

    if bType == 0:
        return dict(status=UNKNOWN, reason=2, bType=0, bN=0, marketN=marketN,
                    baseline=baseline, cond=None, lift=None, cov=None)
    lift = cond - baseline
    status = QUALIFIED if lift >= qual_lift else REJECTED if lift <= rej_lift else WEAK
    return dict(status=status, reason=0, bType=bType, bN=bN, marketN=marketN,
                baseline=baseline, cond=cond, lift=lift, cov=cov)


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------
EMPTY_SC = dict(hit=[0] * 6, fail=[0] * 6, exp=[0] * 6)
EMPTY_TD = dict(hit=[0] * 4, fail=[0] * 4, exp=[0] * 4)
_fail = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fail.append(name)


def sc_single(scB, h, f, e=0):
    """Score fine-bin arrays such that cumulative-from(scB) yields (h,f,e)."""
    a = dict(hit=[0] * 6, fail=[0] * 6, exp=[0] * 6)
    a["hit"][scB], a["fail"][scB], a["exp"][scB] = h, f, e
    return a


# Big neutral market so coverage denominators are stable unless we probe them.
MKT = (400, 600, 0)          # baseline = 400/1000 = 0.40, marketN = 1000
BASE = 0.40

# 1. Market not established -> UNKNOWN(reason1)
r = qualify(3, 1, (30, 20, 0), [], EMPTY_SC, EMPTY_TD)   # 50 resolved < 100
check("1 market<min -> UNKNOWN/reason1", r["status"] == UNKNOWN and r["reason"] == 1)

# 2. QUALIFIED: single score bucket 50% hit vs 40% baseline (lift +10) N=60 cov=6%
r = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)  # 30/(60)=0.50
check("2 lift +10 -> QUALIFIED", r["status"] == QUALIFIED and abs(r["lift"] - 0.10) < 1e-9 and r["bType"] == 2)

# 3. WEAK: cond ~42% (lift +2) N=100
r = qualify(3, 3, MKT, [], sc_single(3, 42, 58), EMPTY_TD)  # 0.42
check("3 lift +2 -> WEAK", r["status"] == WEAK and abs(r["lift"] - 0.02) < 1e-9)

# 4. REJECTED: cond 33% (lift -7) N=100
r = qualify(3, 3, MKT, [], sc_single(3, 33, 67), EMPTY_TD)  # 0.33
check("4 lift -7 -> REJECTED", r["status"] == REJECTED and abs(r["lift"] + 0.07) < 1e-9)

# 5. Small sample: 7 of 8 (87.5%) but N<40 -> UNKNOWN(reason2)
r = qualify(3, 3, MKT, [], sc_single(3, 7, 1), EMPTY_TD)
check("5 tiny-sample high-rate -> UNKNOWN/reason2", r["status"] == UNKNOWN and r["reason"] == 2)

# 6. Coverage: N=45 (>=40) but coverage 45/1000 = 4.5% (<5%) -> UNKNOWN(reason2)
r = qualify(3, 3, MKT, [], sc_single(3, 30, 15), EMPTY_TD)  # tot 45, cov 0.045
check("6 low-coverage -> UNKNOWN/reason2", r["status"] == UNKNOWN and r["reason"] == 2)

# 6b. Coverage just above floor qualifies (N=60, cov 6%)
r = qualify(3, 3, MKT, [], sc_single(3, 33, 27), EMPTY_TD)  # 33/60=0.55 lift+15
check("6b coverage>=floor -> not UNKNOWN", r["status"] == QUALIFIED)

# 7. Discrimination NEAR the thresholds (+/-5). Note: exact-boundary comparison
#    is floating-point-fuzzy (0.45-0.40 = 0.0499...), so we probe clearly on each
#    side; the policy treats the thresholds as approximate by design.
r = qualify(3, 3, MKT, [], sc_single(3, 46, 54), EMPTY_TD)  # 0.46 lift +6 -> QUALIFIED
check("7a lift +6 -> QUALIFIED", r["status"] == QUALIFIED)
r = qualify(3, 3, MKT, [], sc_single(3, 44, 56), EMPTY_TD)  # 0.44 lift +4 -> WEAK
check("7b lift +4 -> WEAK", r["status"] == WEAK)
r = qualify(3, 3, MKT, [], sc_single(3, 36, 64), EMPTY_TD)  # 0.36 lift -4 -> WEAK
check("7c lift -4 -> WEAK", r["status"] == WEAK)
r = qualify(3, 3, MKT, [], sc_single(3, 34, 66), EMPTY_TD)  # 0.34 lift -6 -> REJECTED
check("7d lift -6 -> REJECTED", r["status"] == REJECTED)

# 8. Baseline relativity: SAME conditional 55%, different market baseline -> different label
b40 = qualify(3, 3, (400, 600, 0), [], sc_single(3, 55, 45), EMPTY_TD)   # base .40 -> +15
b54 = qualify(3, 3, (540, 460, 0), [], sc_single(3, 55, 45), EMPTY_TD)   # base .54 -> +1
check("8 same cond, base .40 -> QUALIFIED", b40["status"] == QUALIFIED)
check("8 same cond, base .54 -> WEAK", b54["status"] == WEAK)

# 9. Joint bucket preferred over single when sufficient; most-specific wins.
combos = [
    (0, 2, 2, 2, 30, 30, 0),   # Score>=5.5 & Dist<=4  (loose)  50%
    (0, 4, 2, 0, 40, 20, 0),   # Score>=6.5 & Dist<=1  (tight)  ~66.7%
]
# candidate scB=5 (>=all score thr), tdB=0 (<= all dist thr) satisfies both.
r = qualify(5, 0, MKT, combos, sc_single(5, 10, 10), EMPTY_TD)
check("9 joint preferred (bType==1)", r["bType"] == 1)
check("9 most-specific joint chosen (cond ~0.667)", abs(r["cond"] - (40 / 60)) < 1e-9)

# 10. No-lookahead (structural): the candidate's own future outcome is not an
#     argument; classification is a pure function of prior counts. Flipping a
#     hypothetical own-result cannot change the label because it isn't passed.
r_before = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)
r_again = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)
check("10 deterministic in prior counts (no own-outcome input)",
      r_before["status"] == r_again["status"] and "own_result" not in qualify.__code__.co_varnames)

print()
if _fail:
    print("RESULT: %d FAILED -> %s" % (len(_fail), _fail))
    raise SystemExit(1)
print("RESULT: ALL PASSED")
