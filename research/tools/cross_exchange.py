#!/usr/bin/env python3
"""Borsalar arası fark / TL primi ÖLÇÜM aracı — sadece gözlemler, işlem YAPMAZ.

Paribu'nun BTC_TL fiyatını, Binance'in global fiyatının TL karşılığıyla
karşılaştırır ve aradaki **TL primini** ölçer.

    TL primi = (Paribu_BTC_TL / global_BTC_TL_karsiligi - 1) * 100
    global_BTC_TL_karsiligi = BTCUSDT * USDTTRY   (veya dogrudan BTCTRY)

**BU BİR GÖZLEM ARACIDIR, İŞLEM STRATEJİSİ DEĞİLDİR.**

Ölçülen fark **tek borsayla yakalanamaz**. Bunu paraya çevirmek için:

* iki borsada da **aynı anda bakiye** tutmak (biri TL, diğeri USDT/BTC),
* transfer süresini ve zincir/çekim ücretlerini göze almak,
* transfer sırasında primin kapanma riskini üstlenmek,
* her iki tarafta komisyon + spread ödemek gerekir.

Yani buradaki "%X prim" rakamı bir **kâr değil, bir gözlemdir**. Prim TL'nin
konvertibilite maliyetini, sermaye kontrollerini ve yerel talebi yansıtır;
kalıcı olması normaldir ve kalıcı olması onu ücretsiz para yapmaz.

Anahtar/HMAC yoktur; yalnızca public fiyat okunur, ``submit_order`` çağrılmaz.

Kullanım::

    python -m research.tools.cross_exchange
    python -m research.tools.cross_exchange --watch 600 --interval 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import CONFIG

PARIBU_TICKER_URL = "https://www.paribu.com/ticker"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

#: Global fiyatı TL'ye çevirmek için denenecek yollar (tercih sırasıyla).
#: Doğrudan BTCTRY varsa en temizi odur; yoksa BTCUSDT × USDTTRY kullanılır.
BINANCE_DIRECT = "BTCTRY"
BINANCE_LEG_A = "BTCUSDT"
BINANCE_LEG_B = "USDTTRY"


def _log(msg: str) -> None:
    """Zaman damgalı çıktı (UTC)."""
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def _to_float(value: Any) -> float:
    """Sayıyı sessizce bozmadan float'a çevirir (muhafazakâr iki aşamalı).

    Args:
        value: Ham değer.

    Returns:
        Float değer; çevrilemezse ``nan``.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# Ağ (ince katman — testlerde enjekte edilir)
# --------------------------------------------------------------------------- #


def _http_json(url: str, timeout: float = 15.0) -> Any:
    """Tek bir GET isteği atıp JSON döndürür."""
    import ssl
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "paribu-research-xexch/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return json.loads(r.read().decode("utf-8"))


@dataclass
class PriceSources:
    """Fiyat kaynakları — testlerde tamamen enjekte edilir.

    Attributes:
        paribu: Paribu ticker yanıtını döndüren çağrılabilir.
        binance: Binance sembolü alıp fiyat (float) döndüren çağrılabilir.
            Sembol yoksa ``None`` döndürmelidir.
    """

    paribu: Callable[[], dict[str, Any]] = field(
        default_factory=lambda: (lambda: _http_json(PARIBU_TICKER_URL))
    )
    binance: Callable[[str], float | None] = field(
        default_factory=lambda: _binance_price
    )


def _binance_price(symbol: str) -> float | None:
    """Binance public fiyat ucundan tek sembolün fiyatını çeker.

    Args:
        symbol: Binance sembolü (örn. ``"BTCUSDT"``).

    Returns:
        Fiyat veya sembol yoksa ``None``.
    """
    try:
        data = _http_json(f"{BINANCE_PRICE_URL}?symbol={symbol}")
    except Exception:  # noqa: BLE001 - sembol yoksa 400 doner; bu bir olgudur
        return None
    price = _to_float(data.get("price")) if isinstance(data, dict) else float("nan")
    return price if price > 0 else None


# --------------------------------------------------------------------------- #
# Ölçüm (saf çekirdek)
# --------------------------------------------------------------------------- #


def paribu_btc_try(payload: dict[str, Any], symbol: str = "BTC_TL") -> float:
    """Paribu ticker yanıtından BTC_TL son fiyatını çıkarır.

    Args:
        payload: Ticker yanıtı.
        symbol: Parite adı.

    Returns:
        Son fiyat; bulunamazsa ``nan``.
    """
    entry = payload.get(symbol)
    if not isinstance(entry, dict):
        return float("nan")
    return _to_float(entry.get("last"))


def global_btc_try(
    binance: Callable[[str], float | None],
) -> tuple[float, dict[str, Any]]:
    """Binance'ten global BTC fiyatının TL karşılığını hesaplar.

    Önce doğrudan ``BTCTRY`` denenir (varsa en temiz ölçüm). Yoksa
    ``BTCUSDT × USDTTRY`` çarpımı kullanılır.

    Args:
        binance: Sembol -> fiyat çağrılabiliri.

    Returns:
        ``(fiyat, detay)``. Hesaplanamazsa fiyat ``nan``.
    """
    direct = binance(BINANCE_DIRECT)
    if direct is not None and direct > 0:
        return direct, {"yol": "dogrudan", "semboller": {BINANCE_DIRECT: direct}}

    a = binance(BINANCE_LEG_A)
    b = binance(BINANCE_LEG_B)
    if a and b and a > 0 and b > 0:
        return a * b, {"yol": "carpim", "semboller": {BINANCE_LEG_A: a, BINANCE_LEG_B: b}}
    return float("nan"), {
        "yol": "yok",
        "semboller": {BINANCE_DIRECT: direct, BINANCE_LEG_A: a, BINANCE_LEG_B: b},
    }


def compute_premium(paribu_price: float, global_price: float) -> float:
    """TL primini yüzde olarak hesaplar.

    Args:
        paribu_price: Paribu BTC_TL fiyatı.
        global_price: Global BTC fiyatının TL karşılığı.

    Returns:
        Prim yüzdesi; girdiler geçersizse ``nan``.
    """
    if not (paribu_price > 0 and global_price > 0):
        return float("nan")
    return (paribu_price / global_price - 1.0) * 100.0


def measure_once(sources: PriceSources | None = None) -> dict[str, Any]:
    """Tek bir anlık prim ölçümü yapar.

    Args:
        sources: Fiyat kaynakları (test için enjekte edilebilir).

    Returns:
        ``paribu``, ``global``, ``prim_yuzde``, ``yol``, ``semboller``,
        ``zaman`` ve ``gecerli`` alanlı sözlük.
    """
    s = sources or PriceSources()
    payload = s.paribu()
    p = paribu_btc_try(payload)
    g, detail = global_btc_try(s.binance)
    prem = compute_premium(p, g)
    return {
        "zaman": datetime.now(timezone.utc),
        "paribu": p,
        "global": g,
        "prim_yuzde": prem,
        "yol": detail["yol"],
        "semboller": detail["semboller"],
        "gecerli": bool(prem == prem),  # NaN degilse gecerli
    }


def summarize(samples: list[float]) -> dict[str, float]:
    """Prim örneklerinin dağılımını özetler.

    Tek bir anlık prim yanıltıcıdır: önemli olan primin ne kadar BÜYÜK ve ne
    kadar KALICI olduğudur. Standart sapma, primin dalgalanıp dalgalanmadığını
    (yani kapanma eğiliminde olup olmadığını) gösterir.

    Args:
        samples: Prim yüzdeleri.

    Returns:
        ``n``, ``ortalama``, ``std``, ``min``, ``maks``, ``medyan``,
        ``pozitif_oran``.
    """
    vals = [v for v in samples if v == v]
    if not vals:
        return {"n": 0.0}
    return {
        "n": float(len(vals)),
        "ortalama": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "maks": max(vals),
        "medyan": statistics.median(vals),
        "pozitif_oran": sum(1 for v in vals if v > 0) / len(vals),
    }


def watch(
    duration: float,
    interval: float,
    sources: PriceSources | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    verbose: bool = True,
) -> dict[str, Any]:
    """Belirli süre boyunca primi tekrar tekrar ölçer.

    Args:
        duration: İzleme süresi (saniye).
        interval: Ölçümler arası bekleme.
        sources: Fiyat kaynakları.
        sleeper: Enjekte edilebilir bekleme (test için).
        clock: Enjekte edilebilir saat (test için).
        verbose: Her ölçümü logla.

    Returns:
        ``ozet`` (dağılım) ve ``hata`` sayısı.
    """
    t0 = clock()
    samples: list[float] = []
    errors = 0

    while clock() - t0 < duration:
        try:
            m = measure_once(sources)
        except Exception as e:  # noqa: BLE001 - izleme dongusu dusmemeli
            errors += 1
            _log(f"olcum hatasi: {type(e).__name__}: {e}")
            sleeper(interval)
            continue
        if m["gecerli"]:
            samples.append(m["prim_yuzde"])
            if verbose:
                _log(f"Paribu {m['paribu']:,.0f}  global {m['global']:,.0f}  "
                     f"prim {m['prim_yuzde']:+.3f}%")
        else:
            errors += 1
            if verbose:
                _log(f"gecersiz olcum (yol={m['yol']}, semboller={m['semboller']})")
        sleeper(interval)

    return {"ozet": summarize(samples), "hata": errors, "ornekler": samples}


# --------------------------------------------------------------------------- #
# Raporlama
# --------------------------------------------------------------------------- #


DISCLAIMER = """\
================================================================================
BU BIR GOZLEM/SINYAL ARACIDIR — ISLEM STRATEJISI DEGILDIR.
================================================================================
Olculen TL primi TEK BORSAYLA YAKALANAMAZ. Paraya cevirmek icin iki borsada da
ayni anda bakiye tutmak, transfer suresi + cekim ucretlerini gozetmek, transfer
sirasinda primin kapanma riskini ustlenmek ve her iki tarafta komisyon + spread
odemek gerekir. Prim, TL'nin konvertibilite maliyetini ve yerel talebi yansitir;
kalici olmasi normaldir ve kalici olmasi onu ucretsiz para YAPMAZ.

Bu arac yalnizca public fiyat okur. Emir gondermez, anahtar kullanmaz.
================================================================================
"""


def print_measurement(m: dict[str, Any]) -> None:
    """Tek ölçümü basar."""
    print("\n--- ANLIK OLCUM ---")
    if not m["gecerli"]:
        print(f"  Olculemedi. Yol: {m['yol']}, semboller: {m['semboller']}")
        print("  Paribu veya Binance fiyati alinamadi (ag erisimi / sembol yok).")
        return
    print(f"  Paribu BTC_TL      : {m['paribu']:>16,.2f} TL")
    print(f"  Global BTC (TL)    : {m['global']:>16,.2f} TL   (yol: {m['yol']})")
    for sym, px in m["semboller"].items():
        if px:
            print(f"      {sym:<10} {px:>16,.6f}")
    print(f"  TL PRIMI           : {m['prim_yuzde']:>+15.3f}%")
    fee = CONFIG.backtest.commission_rate * 100
    print(f"\n  Kiyas: tek yon komisyon {fee:.2f}%; iki borsada gidis-donus + transfer")
    print(f"  ucretleri toplami tipik olarak %1'in uzerindedir. Prim bunun altindaysa")
    print(f"  islem maliyetiyle zaten kapanir.")


def print_watch(w: dict[str, Any]) -> None:
    """İzleme özetini basar."""
    s = w["ozet"]
    print("\n" + "=" * 78)
    print("PRIM IZLEME OZETI")
    print("=" * 78)
    if s.get("n", 0) == 0:
        print(f"  Hic gecerli olcum yapilamadi. (hata: {w['hata']})")
        return
    print(f"  Ornek sayisi   : {s['n']:.0f}   (hata: {w['hata']})")
    print(f"  Ortalama prim  : {s['ortalama']:+.3f}%")
    print(f"  Std sapma      : {s['std']:.3f}%")
    print(f"  Min / Maks     : {s['min']:+.3f}%  /  {s['maks']:+.3f}%")
    print(f"  Medyan         : {s['medyan']:+.3f}%")
    print(f"  Pozitif oran   : {s['pozitif_oran']:.1%}")
    print()
    if s["std"] < abs(s["ortalama"]) / 4:
        print("  Yorum: prim DALGALANMIYOR, kalici bir seviye gibi duruyor.")
        print("  Kalici prim arbitraj firsati DEGILDIR — piyasanin TL'ye bicti-")
        print("  gi yapisal fiyattir (konvertibilite maliyeti, yerel talep).")
    else:
        print("  Yorum: prim dalgalaniyor. Dalgalanma, primin ortalamaya donme")
        print("  egilimi olabilecegini dusundurur; ancak bunu paraya cevirmek")
        print("  yine de iki borsada bakiye ve transfer riski gerektirir.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Komut satırı giriş noktası.

    Args:
        argv: Argümanlar (test için).

    Returns:
        Çıkış kodu.
    """
    ap = argparse.ArgumentParser(
        description="Paribu / Binance TL primi OLCUM araci (islem yapmaz, anahtarsiz).",
    )
    ap.add_argument("--watch", type=float, default=None, help="Izleme suresi (saniye).")
    ap.add_argument("--interval", type=float, default=10.0, help="Olcumler arasi bekleme (sn).")
    ap.add_argument("--quiet", action="store_true", help="Izlemede her olcumu loglama.")
    args = ap.parse_args(argv)

    print(DISCLAIMER)

    sources = PriceSources()
    try:
        m = measure_once(sources)
    except Exception as e:  # noqa: BLE001
        print(f"Olcum yapilamadi: {type(e).__name__}: {e}")
        print("Olasi nedenler: ag erisimi kapali, rate limit, uc degismis.")
        return 1

    print_measurement(m)
    if not m["gecerli"]:
        return 1

    if args.watch:
        print(f"\n{args.watch:.0f} saniye boyunca {args.interval:.0f}s araliklarla izleniyor...\n")
        w = watch(args.watch, args.interval, sources=sources, verbose=not args.quiet)
        print_watch(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
