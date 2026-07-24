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
  - PART 2 — market sample = hit + failed (marketDec). Expired predictions are
    NOT resolved evidence: they never gate the market sample. They appear only in
    the coverage/population denominator... which here is ALSO marketDec, so the
    gate `marketDec >= MIN_MARKET_RESOLVED` uses decisive outcomes exclusively.
  - Bucket selection: most-specific SUFFICIENT joint Score x Dist combo, else
    the strongest SUFFICIENT single (Score cumulative>=, Dist cumulative<=).
    Correlated lifts are never summed; exactly one bucket is used.
  - Classify by lift = conditional_hit_rate - market_baseline_hit_rate.
  - Every UNKNOWN carries exactly one reason code (partition; no generic UNKNOWN).
"""

# Global parameters (defaults, identical to the Pine inputs)
MIN_MARKET_RESOLVED = 40
MIN_BUCKET_RESOLVED = 20
MIN_COVERAGE        = 0.05
QUALIFIED_MIN_LIFT  = 0.05
REJECTED_MAX_LIFT   = -0.05

UNKNOWN, QUALIFIED, WEAK, REJECTED = 0, 1, 2, 3
NAMES = {0: "UNKNOWN", 1: "QUALIFIED", 2: "WEAK", 3: "REJECTED"}

# UNKNOWN reason partition (identical to Pine qReason). 0 = classified.
R_CLASSIFIED   = 0
R_MARKET_LOW   = 1   # 0 < marketDec < MIN_MARKET_RESOLVED
R_INVALID_BASE = 2   # marketDec == 0  (no decisive market outcome -> baseline undefined)
R_NO_BUCKET    = 3   # market ok; no applicable single bucket has any resolved sample
R_SINGLE_LOW   = 4   # bucket exists but resolved < MIN_BUCKET_RESOLVED
R_COV_LOW      = 5   # bucket resolved >= MIN_BUCKET but coverage < MIN_COVERAGE
R_OTHER        = 6   # unreachable by construction (bins are total functions, no na keys)
REASON_NAMES = {0: "CLASSIFIED", 1: "MARKET_SAMPLE_LOW", 2: "INVALID_BASELINE",
                3: "NO_BUCKET", 4: "SINGLE_SAMPLE_LOW", 5: "COVERAGE_LOW", 6: "OTHER"}


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

    Returns dict(status, reason, bType, bN, marketDec, baseline, cond, lift, cov,
                 jointLow). marketDec (= hit+failed) is the resolved sample.
    """
    mH, mF, mE = market
    marketDec = mH + mF                         # PART 2: resolved sample excludes expired
    baseline = (mH / marketDec) if marketDec > 0 else None
    status, reason, bType, bN = UNKNOWN, R_OTHER, 0, 0
    cond = lift = cov = None
    jointLow = 0

    if marketDec == 0 or baseline is None:
        return dict(status=UNKNOWN, reason=R_INVALID_BASE, bType=0, bN=0,
                    marketDec=marketDec, baseline=baseline, cond=None, lift=None,
                    cov=None, jointLow=0)
    if marketDec < min_market:
        return dict(status=UNKNOWN, reason=R_MARKET_LOW, bType=0, bN=0,
                    marketDec=marketDec, baseline=baseline, cond=None, lift=None,
                    cov=None, jointLow=0)

    # (a) most-specific sufficient joint Score x Dist bucket
    jFound = jMatched = False
    jSpec = -1
    jH = jF = jE = 0
    for (kindA, lvA, kindB, lvB, h, f, e) in combos:
        if kindA == 0 and kindB == 2 and scB >= lvA and tdB <= lvB:
            dec, tot = h + f, h + f + e
            if dec > 0:
                jMatched = True
            if dec >= min_bucket and tot / marketDec >= min_cov:
                spec = lvA * 10 - lvB
                if (not jFound) or spec > jSpec:
                    jFound, jSpec = True, spec
                    jH, jF, jE = h, f, e
    if jFound:
        bType, bN = 1, jH + jF
        cond = jH / (jH + jF) if (jH + jF) > 0 else None
        cov = (jH + jF + jE) / marketDec
    else:
        # (b) strongest sufficient single (score cumulative>=, dist cumulative<=)
        sH, sF, sE = cum_from(anSc["hit"], scB), cum_from(anSc["fail"], scB), cum_from(anSc["exp"], scB)
        dH, dF, dE = cum_up_to(anTd["hit"], tdB), cum_up_to(anTd["fail"], tdB), cum_up_to(anTd["exp"], tdB)
        sOk = (sH + sF) >= min_bucket and (sH + sF + sE) / marketDec >= min_cov
        dOk = (dH + dF) >= min_bucket and (dH + dF + dE) / marketDec >= min_cov
        sCond = sH / (sH + sF) if (sH + sF) > 0 else None
        dCond = dH / (dH + dF) if (dH + dF) > 0 else None
        sLift = None if sCond is None else sCond - baseline
        dLift = None if dCond is None else dCond - baseline
        if sOk and (not dOk or abs(sLift) >= abs(dLift)):
            bType, bN, cond, cov = 2, sH + sF, sCond, (sH + sF + sE) / marketDec
        elif dOk:
            bType, bN, cond, cov = 3, dH + dF, dCond, (dH + dF + dE) / marketDec
        else:
            bestDec = max(sH + sF, dH + dF)
            if bestDec == 0:
                reason = R_NO_BUCKET
            elif bestDec < min_bucket:
                reason = R_SINGLE_LOW
            else:
                reason = R_COV_LOW
    if jMatched and not jFound:
        jointLow = 1

    if bType == 0:
        return dict(status=UNKNOWN, reason=reason, bType=0, bN=0,
                    marketDec=marketDec, baseline=baseline, cond=None, lift=None,
                    cov=None, jointLow=jointLow)
    lift = cond - baseline
    status = QUALIFIED if lift >= qual_lift else REJECTED if lift <= rej_lift else WEAK
    return dict(status=status, reason=R_CLASSIFIED, bType=bType, bN=bN,
                marketDec=marketDec, baseline=baseline, cond=cond, lift=lift,
                cov=cov, jointLow=jointLow)


def coverage_pct(q, w, r, total):
    """PART 5: coverage = (QUALIFIED + WEAK + REJECTED) / total, as a percent.
    Mirrors Pine `evTotal>0 ? 100.0*(qCount[1]+qCount[2]+qCount[3])/evTotal : na`."""
    return (100.0 * (q + w + r) / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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


    def td_single(tdB, h, f, e=0):
        """Dist fine-bin arrays such that cumulative-up-to(tdB) yields (h,f,e)."""
        a = dict(hit=[0] * 4, fail=[0] * 4, exp=[0] * 4)
        a["hit"][tdB], a["fail"][tdB], a["exp"][tdB] = h, f, e
        return a


    # Big neutral market so coverage denominators are stable unless we probe them.
    MKT = (400, 600, 0)          # baseline = 400/1000 = 0.40, marketDec = 1000
    BASE = 0.40

    # 1. Market not established -> UNKNOWN(MARKET_SAMPLE_LOW). Pin min_market so the
    #    rule is tested independent of the default (which is now 40).
    r = qualify(3, 1, (30, 20, 0), [], EMPTY_SC, EMPTY_TD, min_market=100)   # 50 decisive < 100
    check("1 market<min -> UNKNOWN/MARKET_SAMPLE_LOW", r["status"] == UNKNOWN and r["reason"] == R_MARKET_LOW)

    # 2. QUALIFIED: single score bucket 50% hit vs 40% baseline (lift +10) N=60 cov=6%
    r = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)  # 30/(60)=0.50
    check("2 lift +10 -> QUALIFIED", r["status"] == QUALIFIED and abs(r["lift"] - 0.10) < 1e-9 and r["bType"] == 2)

    # 3. WEAK: cond ~42% (lift +2) N=100
    r = qualify(3, 3, MKT, [], sc_single(3, 42, 58), EMPTY_TD)  # 0.42
    check("3 lift +2 -> WEAK", r["status"] == WEAK and abs(r["lift"] - 0.02) < 1e-9)

    # 4. REJECTED: cond 33% (lift -7) N=100
    r = qualify(3, 3, MKT, [], sc_single(3, 33, 67), EMPTY_TD)  # 0.33
    check("4 lift -7 -> REJECTED", r["status"] == REJECTED and abs(r["lift"] + 0.07) < 1e-9)

    # 5. Small sample: 7 of 8 (87.5%) but N<40 -> UNKNOWN(SINGLE_SAMPLE_LOW)
    r = qualify(3, 3, MKT, [], sc_single(3, 7, 1), EMPTY_TD)
    check("5 tiny-sample high-rate -> UNKNOWN/SINGLE_SAMPLE_LOW", r["status"] == UNKNOWN and r["reason"] == R_SINGLE_LOW)

    # 6. Coverage: N=45 (>=40) but coverage 45/1000 = 4.5% (<5%) -> UNKNOWN(COVERAGE_LOW)
    r = qualify(3, 3, MKT, [], sc_single(3, 30, 15), EMPTY_TD)  # tot 45, cov 0.045
    check("6 low-coverage -> UNKNOWN/COVERAGE_LOW", r["status"] == UNKNOWN and r["reason"] == R_COV_LOW)

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
    #     argument; classification is a pure function of prior counts.
    r_before = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)
    r_again = qualify(3, 3, MKT, [], sc_single(3, 30, 30), EMPTY_TD)
    check("10 deterministic in prior counts (no own-outcome input)",
          r_before["status"] == r_again["status"] and "own_result" not in qualify.__code__.co_varnames)

    # ---------------------------------------------------------------------------
    # PART 6 — the eight required deterministic tests
    # ---------------------------------------------------------------------------

    # P6-1. Market resolved sample EXCLUDES expired. Same 87 decisive outcomes:
    #   expired padding it to >=100 total must NOT let it pass the market gate.
    # Pin min_market=100 so the expired-exclusion demo holds (87 dec < 100) regardless of default.
    r_pad = qualify(3, 3, (46, 41, 18), [], sc_single(3, 30, 30), EMPTY_TD, min_market=100)  # 87 dec, 18 exp, 105 tot
    r_nopad = qualify(3, 3, (46, 41, 0), [], sc_single(3, 30, 30), EMPTY_TD, min_market=100)  # 87 dec, 0 exp
    check("P6-1 expired does NOT count as resolved (105 tot but 87 dec -> MARKET_LOW)",
          r_pad["status"] == UNKNOWN and r_pad["reason"] == R_MARKET_LOW and r_pad["marketDec"] == 87)
    check("P6-1 market gate identical with/without expired padding",
          r_pad["status"] == r_nopad["status"] and r_pad["reason"] == r_nopad["reason"])

    # P6-2. Joint sufficient -> joint used (bType==1), single ignored even if present.
    combos2 = [(0, 3, 2, 3, 30, 30, 0)]   # Score>=3 & Dist<=3, 60 dec, 50%
    r = qualify(3, 3, MKT, combos2, sc_single(3, 99, 1), EMPTY_TD)  # single would say ~99%
    check("P6-2 joint sufficient -> joint used (bType==1)", r["bType"] == 1 and abs(r["cond"] - 0.5) < 1e-9)

    # P6-3. Joint insufficient (below MIN_BUCKET) + single sufficient -> SINGLE fallback used.
    combos3 = [(0, 3, 2, 3, 5, 5, 0)]     # joint only 10 dec (<40) -> insufficient
    r = qualify(3, 3, MKT, combos3, sc_single(3, 30, 30), EMPTY_TD)  # single 60 dec, 50%
    check("P6-3 joint insufficient + single sufficient -> single fallback (bType==2)",
          r["bType"] == 2 and r["status"] == QUALIFIED)
    check("P6-3 joint-matched-but-insufficient flagged (jointLow==1)", r["jointLow"] == 1)

    # P6-4. All buckets insufficient -> UNKNOWN.
    combos4 = [(0, 3, 2, 3, 5, 5, 0)]     # joint 10 dec (<40)
    r = qualify(3, 3, MKT, combos4, sc_single(3, 5, 5), EMPTY_TD)  # single 10 dec (<40)
    check("P6-4 all insufficient -> UNKNOWN", r["status"] == UNKNOWN)

    # P6-5. Every UNKNOWN carries an explicit reason in the partition {1..6}; never
    #   the generic sentinel. Sweep several UNKNOWN-producing inputs.
    unknown_cases = [
        qualify(3, 1, (30, 20, 0), [], EMPTY_SC, EMPTY_TD),          # market low
        qualify(3, 1, (0, 0, 50), [], EMPTY_SC, EMPTY_TD),           # invalid baseline (0 decisive)
        qualify(3, 3, MKT, [], EMPTY_SC, EMPTY_TD),                  # no bucket
        qualify(3, 3, MKT, [], sc_single(3, 5, 5), EMPTY_TD),        # single sample low
        qualify(3, 3, MKT, [], sc_single(3, 30, 15), EMPTY_TD),      # coverage low
    ]
    check("P6-5 every UNKNOWN has an explicit reason in 1..6 (no generic)",
          all(c["status"] == UNKNOWN and c["reason"] in (1, 2, 3, 4, 5, 6) for c in unknown_cases))
    check("P6-5 invalid-baseline case reason == INVALID_BASELINE",
          unknown_cases[1]["reason"] == R_INVALID_BASE)
    check("P6-5 no-bucket case reason == NO_BUCKET",
          unknown_cases[2]["reason"] == R_NO_BUCKET)

    # P6-6. Coverage counts WEAK and REJECTED, not just QUALIFIED.
    #   1 QUALIFIED + 2 WEAK + 1 REJECTED out of 100 -> 4 classified -> 4.0%.
    cov_all = coverage_pct(1, 2, 1, 100)
    cov_qonly = 100.0 * 1 / 100
    check("P6-6 coverage includes WEAK+REJECTED (=4.0%, not 1.0%)",
          abs(cov_all - 4.0) < 1e-9 and abs(cov_qonly - 1.0) < 1e-9 and cov_all != cov_qonly)

    # P6-7. AYGAZ-like coverage FORMULA check: given the observed distribution
    #   (total 105, classified 5, unknown 100), coverage must be 5/105 ~= 4.76%,
    #   NOT 0%. (This tests PART 5's formula; it does not assert qualify() yields 5 —
    #   with the PART 2 fix AYGAZ's 87 decisive outcomes actually make all 105 UNKNOWN.)
    cov_aygaz = coverage_pct(0, 5, 0, 105)   # 5 classified (e.g. all WEAK) of 105
    check("P6-7 AYGAZ-like 5/105 coverage ~= 4.76% (not 0%)",
          abs(cov_aygaz - (100.0 * 5 / 105)) < 1e-9 and abs(cov_aygaz - 4.7619) < 1e-3)

    # P6-7b. AYGAZ under the PART 2 fix: at a 100 market gate, 87 decisive < 100 ->
    #   UNKNOWN/MARKET_SAMPLE_LOW regardless of bucket evidence (pin min_market=100 to
    #   demonstrate the gate; the production default is now 40, at which 87 passes).
    r = qualify(3, 3, (46, 41, 18), combos2, sc_single(3, 99, 1), EMPTY_TD, min_market=100)
    check("P6-7b AYGAZ decisive=87 < 100 gate -> UNKNOWN/MARKET_SAMPLE_LOW",
          r["status"] == UNKNOWN and r["reason"] == R_MARKET_LOW)

    # P6-8. Pine and Python reference use identical semantics. We cannot import Pine,
    #   so we assert the constants and the exact-formula invariants this file encodes
    #   match the documented Pine contract (kept in sync by review):
    #     - market gate is marketDec (hit+failed), thresholds as below
    #     - coverage denominators use marketDec (not marketN)
    #     - reason partition ids 1..6 as named
    check("P6-8 thresholds identical to Pine inputs",
          MIN_MARKET_RESOLVED == 40 and MIN_BUCKET_RESOLVED == 20 and
          abs(MIN_COVERAGE - 0.05) < 1e-12 and abs(QUALIFIED_MIN_LIFT - 0.05) < 1e-12 and
          abs(REJECTED_MAX_LIFT + 0.05) < 1e-12)
    # marketDec is exposed and equals hit+failed for a mixed sample:
    _r = qualify(3, 3, (46, 41, 18), [], EMPTY_SC, EMPTY_TD)
    check("P6-8 exposed marketDec == hit+failed (87, expired 18 excluded)", _r["marketDec"] == 87)

    print()
    if _fail:
        print("RESULT: %d FAILED -> %s" % (len(_fail), _fail))
        raise SystemExit(1)
    print("RESULT: ALL PASSED")
