# Harmonic Patterns Pro (Open) — Pine Script v6

A TradingView (Pine Script v6) indicator inspired by LonesomeTheBlue's
*Harmonic Patterns Pro*. It detects XABCD harmonic patterns from ZigZag pivots
using Fibonacci ratios with a configurable error tolerance, then draws the
pattern, trade levels and a live statistics table.

File: [`harmonic_patterns_pro.pine`](./harmonic_patterns_pro.pine)

## How it works

1. **ZigZag pivots** — `ta.pivothigh` / `ta.pivotlow` (period = `ZigZag Period`)
   build an alternating high/low pivot sequence. The last 5 pivots are treated
   as **X, A, B, C, D**.
2. **Fibonacci ratios** are measured between the legs:
   - `xab = |B-A| / |X-A|`
   - `abc = |B-C| / |A-B|`
   - `bcd = |C-D| / |B-C|`
   - `xad = |A-D| / |X-A|`
3. Each pattern is matched when all of its ratios fall inside their expected
   Fibonacci windows, widened on both sides by the **Error Rate %** (this is the
   "validation zone" tolerance). A single D point can match several patterns at
   once (e.g. Gartley + Shark + Cypher), exactly like the original.

## Patterns & ratio windows

| Pattern       | xab (AB/XA) | abc (BC/AB) | bcd (CD/BC) | xad (AD/XA) |
|---------------|-------------|-------------|-------------|-------------|
| Gartley       | 0.500–0.618 | 0.382–0.886 | 1.130–2.618 | 0.750–0.875 |
| Butterfly     | 0.700–0.860 | 0.382–0.886 | 1.618–2.618 | 1.270–1.618 |
| Bat           | 0.382–0.500 | 0.382–0.886 | 1.618–2.618 | ≤ 0.886     |
| Alternate Bat | 0.382–0.500 | 0.382–0.886 | 2.000–3.618 | ≤ 1.130     |
| Crab          | 0.382–0.618 | 0.382–0.886 | 2.240–3.618 | ≈ 1.618     |
| Deep Crab     | ≈ 0.886     | 0.382–0.886 | 2.000–3.618 | ≈ 1.618     |
| Shark         | 0.500–0.886 | 1.130–1.618 | 1.270–2.240 | 0.886–1.130 |
| Cypher        | 0.382–0.618 | 1.130–1.414 | 1.272–2.000 | ≈ 0.786     |
| AB=CD         | —           | 0.618–0.786 | 1.272–1.618 | —           |

## Trade levels

Levels are projected from the **A→D** leg (`rng = |A-D|`), configurable in inputs:

- **Entry**  = D
- **Target1** = D ± `rng × 0.382`
- **Target2** = D ± `rng × 0.618`
- **Stop-loss** = D ∓ `rng × 0.236`

(sign depends on bullish/bearish direction). Only the **latest** setup's level
lines and validation-zone box are kept on the chart; pattern geometry (XABCD
lines, letters, ratios) stays for history.

## Statistics table

Every detected setup is tracked bar-by-bar to its outcome and counted per
pattern, split **LONG / SHORT**:

- **Found** — times detected
- **Inv** (invalid) — stop-loss hit before Target1
- **T1 / T2** — reached Target1 / Target2
- **SL** — stop-loss touched

A **Total** row sums each column.

## Inputs

- **ZigZag**: period, error rate %, show/colour zigzag
- **Patterns**: toggle each of the 9 patterns
- **Display**: pattern lines & XABCD, fib ratios, harmonic (XB/BD/AC) lines,
  level lines, validation zone, statistics table + position, level line length
- **Levels**: Target1 / Target2 / Stop-loss fib factors
- **Colours**: one colour per pattern

## Alerts

- Bullish / Bearish / Any harmonic pattern (`alertcondition`)
- A dynamic `alert()` naming the matched pattern(s) once per bar

## Notes & differences from the closed-source original

- Detection uses the **confirmed** ZigZag pivot, which lags by `ZigZag Period`
  bars; the most recent pivot can still extend until an opposite pivot forms
  (inherent ZigZag behaviour). This is not a repainting *trade signal* — a
  detected setup does not move once its D pivot is confirmed.
- This open version ships the 9 core XABCD harmonic patterns. Classic chart
  patterns from the Pro edition (Triangles, Double Top/Bottom, Head & Shoulders,
  Swans, 3-Drive, 5-0, Nen Star, Anti-patterns, etc.) are **not** included yet
  and can be layered on top of the same ZigZag/ratio framework.
- Trade-level and stop models are simplified/parameterised rather than
  per-pattern hard-coded PRZ stops.
