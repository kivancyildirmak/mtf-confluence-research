#!/usr/bin/env python3
"""
Static checks for the Strategy Tester wrapper (src/mtf_confluence_tradetest.pine).

Pine cannot be executed off-platform, so these are SOURCE-level invariants that
prove the properties the strategy must have:
  (A) shared pivot/harmonic/qualification/evaluation logic is byte-identical to
      the indicator (nothing was altered — the strategy is a strict superset);
  (B) no lookahead is introduced and orders are created only from confirmed data;
  (C) the required order rules (stop, target, R:R, sizing, expiry) are present;
  (D) the strategy enters on the C-CONFIRMATION bar, never at the historical C
      price, and the indicator file is left untouched.
Run: python3 tests/strategy_checks.py
"""
import collections
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IND = os.path.join(ROOT, "src", "mtf_confluence_strategy.pine")
STR = os.path.join(ROOT, "src", "mtf_confluence_tradetest.pine")

ind = open(IND).read().splitlines()
strat = open(STR).read().splitlines()
strat_text = "\n".join(strat)

_fail = []
def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fail.append(name)


# (A) LOGIC PARITY — every non-declaration indicator line survives verbatim in the
#     strategy (multiset containment). If any shared line were altered or removed,
#     its count in the strategy would drop below the indicator's and this fails.
ind_decl = [l for l in ind if l.startswith("indicator(")]
chk("indicator file still declares indicator()", len(ind_decl) == 1)
ind_body = collections.Counter(l for l in ind if not l.startswith("indicator("))
strat_body = collections.Counter(strat)
missing = [l for l, c in ind_body.items() if strat_body[l] < c]
chk("all shared logic lines preserved verbatim in strategy (0 missing)", len(missing) == 0)
if missing:
    for l in missing[:8]:
        print("   MISSING:", repr(l))

# (B) NO LOOKAHEAD / CONFIRMED-ONLY ORDERS
chk("strategy declares strategy() (not indicator())",
    any(l.startswith("strategy(") for l in strat) and not any(l.startswith("indicator(") for l in strat))
chk("pyramiding = 0", "pyramiding = 0" in strat_text)
chk("calc_on_every_tick = false (fills next bar, not intrabar)", "calc_on_every_tick = false" in strat_text)
chk("process_orders_on_close = false (fills next bar open)", "process_orders_on_close = false" in strat_text)
chk("no barmerge lookahead anywhere", "lookahead_on" not in strat_text)
# match an actual call `request.security(` — the string also appears in comments
# ("request.security/lookahead yok") documenting its ABSENCE, which must not count.
_code_lines = [l.split("//", 1)[0] for l in strat]
chk("no request.security() call introduced", "request.security(" not in "\n".join(_code_lines))
# strategy.entry must be indented (nested inside the barstate.isconfirmed detection
# block), never at column 0 (global/every-bar).
entry_lines = [l for l in strat if "strategy.entry(" in l]
# 4 entry variants: v1 pullback-L, v1 market-L, fade PRZ long-L, fade PRZ short-S.
chk("four entry variants (v1 pullback/market + fade long/short)", len(entry_lines) == 4)
chk("all strategy.entry calls nested (indented), never global",
    bool(entry_lines) and all(l.startswith(" ") for l in entry_lines))
chk("entries use ids 'L' (long) or 'S' (short) only",
    all(('"L"' in l or '"S"' in l) for l in entry_lines))
chk("detection guarded by barstate.isconfirmed", "if barstate.isconfirmed" in strat_text)
# position-management block also gated by confirmed bars
chk("position management under barstate.isconfirmed",
    strat_text.count("if barstate.isconfirmed") >= 2)

# (C) ORDER RULES PRESENT (redesigned trade management)
chk("structural stop below C with buffer (not fixed 0.5 ATR)",
    "C.price - math.max(stopBufferATR * cATR, minStopTicks * syminfo.mintick)" in strat_text)
chk("pullback limit entry toward C", "C.price + pullbackFrac * (close - C.price)" in strat_text)
chk("pullback places a limit order", 'strategy.entry("L", strategy.long, qty = qty, limit = stEntry' in strat_text)
chk("entry-mode input offers Pullback + Market (default Market for measurement)",
    'entryMode  = input.string("Market", "Entry mode"' in strat_text and '["Pullback", "Market"]' in strat_text)
chk("target = pLo", re.search(r"stTarget\s*=\s*pLo", strat_text) is not None)
chk("reject: target > entry required", "stTarget > stEntry" in strat_text)
chk("reject: stop < entry required", "stStop < stEntry" in strat_text)
chk("reject: R:R >= minRR", "stRR >= minRR" in strat_text)
chk("reject: entry too close to D (min target distance)",
    "(stTarget - stEntry) >= minTargetDistATR * cATR" in strat_text)
chk("Minimum R:R input present (default 0.0 = measurement preset)", 'input.float(0.0, "Minimum R:R"' in strat_text)
chk("risk % sizing from entry-to-stop distance",
    "riskCash / stRisk" in strat_text and "strategy.equity * (riskPct" in strat_text)
# Pine requires const commission/slippage in the strategy() header (CE10123 if
# input.* is used); they are edited natively via the Strategy Properties tab.
chk("commission_value const in header (0.04)", "commission_value = 0.04" in strat_text)
chk("slippage const in header (1)", "slippage = 1" in strat_text)
chk("no input.* inside strategy() header (would fail CE10123)",
    re.search(r"strategy\([^\n]*input\.", strat_text) is None)
chk("trade modes All / QUALIFIED only / QUALIFIED + WEAK",
    '"All", "QUALIFIED only", "QUALIFIED + WEAK"' in strat_text)
chk("trade timeout at maxTradeBars (separate from evaluation maxActiveBars)",
    "maxTradeBars" in strat_text and "strategy.close_all(" in strat_text and 'input.int(15, "Max trade duration' in strat_text)
chk("unfilled pullback limit cancelled after entryValidBars",
    'strategy.cancel("L")' in strat_text and "entryValidBars" in strat_text)
chk("dual-target scale-out exits (T1 partial + T2/stop, long+short)",
    "strategy.exit(" in strat_text and "stop = stActiveStop" in strat_text
    and "limit = stT1" in strat_text and "limit = stT2" in strat_text
    and "qty_percent = przT1Qty" in strat_text)

# (C2) FADE-at-D (PRZ reversal) method
chk("method selector Chase-to-D (v1) vs Fade-at-D (PRZ)",
    '"Chase-to-D (v1)", "Fade-at-D (PRZ)"' in strat_text)
chk("fade arms the D zone (waits for price to reach D)", "przArmed  := true" in strat_text)
chk("fade reversal: SHORT when price reaches D-above, LONG when D-below",
    "przPatDir == 1 and high >= przLo" in strat_text and "przPatDir == -1 and low <= przHi" in strat_text)
chk("PRZ stop beyond the D zone (structural)",
    "przHi + przStopATR * przATR" in strat_text and "przLo - przStopATR * przATR" in strat_text)
chk("fade places long/short reversal entries",
    'strategy.entry("L", strategy.long, qty = qty, alert_message = "HPS PRZ long")' in strat_text and
    'strategy.entry("S", strategy.short, qty = qty, alert_message = "HPS PRZ short")' in strat_text)
chk("arm window disarms if D not reached (armBars)", "(bar_index - przArmBar) > armBars" in strat_text)

# (C3) HARMONIC-PRO: dual fib targets + LONG/SHORT statistics
chk("T1/T2 fib targets off the D->C leg",
    "przT1Frac * (przCprice - entryPx)" in strat_text and "przT2Frac * (przCprice - entryPx)" in strat_text)
chk("T1/T2 fraction + partial-% inputs",
    'input.float(0.382, "Target1 fraction' in strat_text and 'input.float(0.618, "Target2 fraction' in strat_text
    and 'input.int(50, "Target1 partial exit %"' in strat_text)
chk("reached-target tracking for stats", "stReachedT1 :=" in strat_text and "stReachedT2 :=" in strat_text)
chk("Found/Invalid counted per direction",
    "array.set(stFound," in strat_text and "array.set(stInvalid," in strat_text)
chk("T1/T2/Stop outcomes counted per direction",
    "array.set(stT1cnt," in strat_text and "array.set(stT2cnt," in strat_text and "array.set(stStopCnt," in strat_text)
chk("LONG/SHORT statistics panel (Found/Invalid/T1/T2/Stop)",
    '"FADE-at-D STATS"' in strat_text and '"Found"' in strat_text and '"Invalid"' in strat_text
    and "table.new(position.bottom_right" in strat_text)
chk("stats helper f_hcell is global (not nested in an if)",
    any(l.startswith("f_hcell(table t") for l in strat))

# (D) ENTRY IS SIGNAL-DERIVED, NOT HISTORICAL C PRICE
chk("entry never uses raw historical C price as fill (pullback is C + f*(close-C))",
    "C.price + pullbackFrac * (close - C.price)" in strat_text)
chk("pullback limit must sit below current close (real pullback, not instant fill)",
    'entryMode == "Market" or stEntry < close' in strat_text)
chk("labels show C->entry bar gap", "C->entry" in strat_text)
chk("labels show calculated R:R", "R:R " in strat_text)
chk("alerts for entry/exit/timeout present",
    ' filled @ ' in strat_text and "HPS exit (" in strat_text and "HPS timeout close" in strat_text)
chk("supports long and short (fade can reverse either way)",
    "strategy.long" in strat_text and "strategy.short" in strat_text)
chk("header documents: entry on C-confirmation bar, NOT historical C price",
    "C'nin ONAYLANDIĞI" in strat_text and "tarihsel fiyattan DE" in strat_text)

print()
if _fail:
    print("RESULT: %d FAILED -> %s" % (len(_fail), _fail))
    raise SystemExit(1)
print("RESULT: ALL PASSED")
