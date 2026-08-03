#!/usr/bin/env python3
"""Paribu içi ÜÇGEN ARBİTRAJ ölçüm aracı — sadece ölçer, işlem YAPMAZ.

Üçgen arbitraj: aynı borsada üç işlemle başladığın paraya geri dönmek
(TL -> BTC -> USDT -> TL). Fiyatlar bir an için tutarsızsa döngü sonunda
başladığından fazlası kalır.

**BU ARAÇ İŞLEM YAPMAZ.** Yalnızca public ticker'ı okur ve "şu anda net kârlı
bir döngü var mı?" sorusunu ölçer. Anahtar/HMAC yoktur, ``submit_order``
çağrılmaz.

**GERÇEKÇİ FİYATLAMA.** Orta fiyat (mid) kullanmak üçgen arbitrajda en yaygın
hatadır ve olmayan fırsatları var gösterir. Burada her bacak, işlemin gerçekte
dolacağı taraftan fiyatlanır:

* Bir coin **alıyorsan** ``lowestAsk`` ödersin (satıcının istediği fiyat),
* Bir coin **satıyorsan** ``highestBid`` alırsın (alıcının verdiği fiyat).

Aradaki fark (spread) her bacakta aleyhine çalışır; üç bacakta üç kez.

**KOMİSYON ZORUNLU.** Üç işlem × işlem başına komisyon (varsayılan %0.2)
= toplam ~%0.6. Brüt getiri bunun altındaysa döngü zarardadır. Maliyetsiz
üçgen arbitraj hesabı her zaman "fırsat" gösterir ve tamamen sahtedir.

**ÜÇGEN İÇİN COIN-COIN PARİTESİ ŞARTTIR.** Borsada yalnızca ``X_TL`` türü
pariteler varsa (her şey TL'ye karşı), üçgen kurulamaz — çünkü orta bacak
(coin'den coin'e) yoktur. Araç önce hangi paritelerin var olduğunu raporlar.

Kullanım::

    python -m research.tools.triangular_paribu
    python -m research.tools.triangular_paribu --watch 300 --interval 5
    python -m research.tools.triangular_paribu --list-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import permutations
from typing import Any, Callable, Iterable

from ..config import CONFIG

#: Paribu public ticker ucu (probe ile doğrulanmış tek çalışan uç).
TICKER_URL = "https://www.paribu.com/ticker"

#: Bir bacakta kullanılabilir sayılması için fiyatın pozitif olması gerekir.
MIN_PRICE = 1e-12


def _log(msg: str) -> None:
    """Zaman damgalı çıktı (UTC)."""
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def _to_float(value: Any) -> float:
    """Sayıyı sessizce bozmadan float'a çevirir.

    :func:`research.tools.collect_paribu._to_float` ile aynı muhafazakâr
    sıralamayı kullanır: önce standart ``float()``, yalnızca o başarısız olursa
    TR biçimi. Ters sıra ``"3000.50"`` -> ``300050`` gibi sessiz bir bozulma
    yaratırdı.

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


def fetch_ticker(getter: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Paribu public ticker'ını çeker.

    Args:
        getter: Test için enjekte edilebilen veri sağlayıcı. ``None`` ise
            gerçek HTTP isteği atılır.

    Returns:
        Parite -> alanlar sözlüğü.

    Raises:
        Exception: Ağ/ayrıştırma hatalarında (çağıran yakalar).
    """
    if getter is not None:
        return getter()
    import ssl
    import urllib.request

    req = urllib.request.Request(
        TICKER_URL,
        headers={"User-Agent": "paribu-research-triarb/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15.0, context=ssl.create_default_context()) as r:
        return json.loads(r.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Piyasa grafiği (saf fonksiyonlar — ağsız test edilir)
# --------------------------------------------------------------------------- #


def parse_symbol(symbol: str) -> tuple[str, str] | None:
    """``"BTC_TL"`` gibi bir sembolü ``(baz, kote)`` ikilisine ayırır.

    Args:
        symbol: Parite adı.

    Returns:
        ``(baz, kote)`` veya biçim tanınmıyorsa ``None``.
    """
    if not isinstance(symbol, str) or "_" not in symbol:
        return None
    base, _, quote = symbol.partition("_")
    base, quote = base.strip().upper(), quote.strip().upper()
    if not base or not quote or base == quote:
        return None
    return base, quote


@dataclass(frozen=True)
class Market:
    """Tek bir paritenin işlem yapılabilir fiyatları.

    Attributes:
        symbol: Parite adı.
        base: Baz varlık.
        quote: Kote varlık.
        ask: ``lowestAsk`` — baz varlığı ALMAK için ödenen fiyat.
        bid: ``highestBid`` — baz varlığı SATMAK için alınan fiyat.
    """

    symbol: str
    base: str
    quote: str
    ask: float
    bid: float

    @property
    def spread_rel(self) -> float:
        """Göreli spread (mid'e oranla)."""
        mid = (self.ask + self.bid) / 2.0
        return (self.ask - self.bid) / mid if mid > 0 else float("nan")


def build_markets(payload: dict[str, Any]) -> dict[tuple[str, str], Market]:
    """Ticker yanıtından ``(baz, kote) -> Market`` haritası kurar.

    Ask/bid'i olmayan veya bozuk pariteler sessizce ATLANMAZ; yalnızca
    kullanılamaz oldukları için haritaya girmezler ve
    :func:`describe_universe` tarafından raporlanırlar.

    Args:
        payload: :func:`fetch_ticker` çıktısı.

    Returns:
        Piyasa haritası.
    """
    markets: dict[tuple[str, str], Market] = {}
    for symbol, data in payload.items():
        parsed = parse_symbol(symbol)
        if parsed is None or not isinstance(data, dict):
            continue
        base, quote = parsed
        ask = _to_float(data.get("lowestAsk"))
        bid = _to_float(data.get("highestBid"))
        if not (ask > MIN_PRICE and bid > MIN_PRICE):
            continue
        markets[(base, quote)] = Market(symbol, base, quote, ask, bid)
    return markets


def describe_universe(payload: dict[str, Any]) -> dict[str, Any]:
    """Borsadaki parite evrenini raporlar — üçgen mümkün mü, önce bu.

    Args:
        payload: Ticker yanıtı.

    Returns:
        ``tum_semboller``, ``gecerli``, ``kullanilamaz``, ``para_birimleri``,
        ``coin_coin`` (kote tarafı TL olmayan pariteler) ve ``ucgen_mumkun``.

    Notes:
        ``coin_coin`` listesi BOŞSA üçgen arbitraj **yapısal olarak imkânsızdır**:
        her şey TL'ye karşı işlem görüyorsa TL -> X -> Y -> TL döngüsünün orta
        bacağı (X_Y paritesi) yoktur.
    """
    all_symbols = sorted(str(s) for s in payload)
    markets = build_markets(payload)
    usable = sorted(m.symbol for m in markets.values())
    unusable = sorted(set(all_symbols) - set(usable))
    currencies = sorted({c for pair in markets for c in pair})
    fiat = {"TL", "TRY"}
    coin_coin = sorted(
        m.symbol for m in markets.values() if m.base not in fiat and m.quote not in fiat
    )
    return {
        "tum_semboller": all_symbols,
        "gecerli": usable,
        "kullanilamaz": unusable,
        "para_birimleri": currencies,
        "coin_coin": coin_coin,
        "ucgen_mumkun": len(coin_coin) > 0,
    }


def convert(
    amount: float,
    from_cur: str,
    to_cur: str,
    markets: dict[tuple[str, str], Market],
    fee: float,
) -> tuple[float, dict[str, Any]] | None:
    """Bir bacağı GERÇEKÇİ yönde fiyatlayarak dönüştürür.

    Args:
        amount: Elde bulunan ``from_cur`` miktarı.
        from_cur: Kaynak varlık.
        to_cur: Hedef varlık.
        markets: Piyasa haritası.
        fee: İşlem başına komisyon oranı.

    Returns:
        ``(yeni_miktar, bacak_detayi)`` veya bu bacak için piyasa yoksa ``None``.

    Notes:
        İki durum vardır:

        * ``(to_cur, from_cur)`` paritesi varsa hedef varlığı **alıyoruz**:
          ``lowestAsk`` öderiz, miktar ``amount / ask`` olur.
        * ``(from_cur, to_cur)`` paritesi varsa kaynak varlığı **satıyoruz**:
          ``highestBid`` alırız, miktar ``amount * bid`` olur.

        Komisyon her iki durumda da alınan miktardan düşülür.
    """
    m = markets.get((to_cur, from_cur))
    if m is not None:  # to_cur'u satın alıyoruz -> ask öde
        out = amount / m.ask * (1.0 - fee)
        return out, {"yon": "AL", "parite": m.symbol, "fiyat": m.ask,
                     "fiyat_tipi": "lowestAsk", "girdi": amount, "cikti": out}
    m = markets.get((from_cur, to_cur))
    if m is not None:  # from_cur'u satıyoruz -> bid al
        out = amount * m.bid * (1.0 - fee)
        return out, {"yon": "SAT", "parite": m.symbol, "fiyat": m.bid,
                     "fiyat_tipi": "highestBid", "girdi": amount, "cikti": out}
    return None


def find_cycles(
    markets: dict[tuple[str, str], Market],
    start: str | None = None,
) -> list[tuple[str, str, str]]:
    """Mümkün tüm üçgen döngülerini bulur.

    Döngünün dönüşleri (rotasyonları) aynı getiriyi verdiği için tekilleştirilir;
    ters yön ise AYRI bir döngüdür ve korunur (getirisi farklıdır).

    Args:
        markets: Piyasa haritası.
        start: Yalnızca bu varlıkla başlayan döngüler (örn. ``"TL"``).

    Returns:
        ``(a, b, c)`` üçlüleri listesi.
    """
    currencies = sorted({c for pair in markets for c in pair})

    def linked(x: str, y: str) -> bool:
        return (y, x) in markets or (x, y) in markets

    seen: set[tuple[str, str, str]] = set()
    cycles: list[tuple[str, str, str]] = []
    for a, b, c in permutations(currencies, 3):
        rotations = [(a, b, c), (b, c, a), (c, a, b)]
        canonical = min(rotations)
        if canonical in seen:
            continue
        if not (linked(a, b) and linked(b, c) and linked(c, a)):
            continue
        seen.add(canonical)
        if start is not None:
            # Döngüyü istenen varlıktan başlayacak şekilde döndür.
            for rot in rotations:
                if rot[0] == start.upper():
                    cycles.append(rot)
                    break
        else:
            cycles.append(canonical)
    return cycles


def evaluate_cycle(
    cycle: tuple[str, str, str],
    markets: dict[tuple[str, str], Market],
    fee: float,
) -> dict[str, Any] | None:
    """Tek bir döngünün brüt ve net getirisini hesaplar.

    Args:
        cycle: ``(a, b, c)`` varlık üçlüsü.
        markets: Piyasa haritası.
        fee: İşlem başına komisyon.

    Returns:
        Döngü sonucu sözlüğü veya bacaklardan biri yoksa ``None``.
    """
    a, b, c = cycle
    path = [a, b, c, a]

    def walk(f: float) -> tuple[float, list[dict[str, Any]]] | None:
        amount = 1.0
        legs: list[dict[str, Any]] = []
        for i in range(3):
            step = convert(amount, path[i], path[i + 1], markets, f)
            if step is None:
                return None
            amount, leg = step
            legs.append(leg)
        return amount, legs

    net_walk = walk(fee)
    gross_walk = walk(0.0)
    if net_walk is None or gross_walk is None:
        return None

    net_amount, legs = net_walk
    gross_amount, _ = gross_walk
    return {
        "dongu": " -> ".join(path),
        "baslangic": a,
        "brut_getiri": gross_amount - 1.0,
        "net_getiri": net_amount - 1.0,
        "toplam_komisyon": 3.0 * fee,
        "kar_var": net_amount > 1.0,
        "bacaklar": legs,
    }


def measure(
    payload: dict[str, Any],
    fee: float | None = None,
    start: str | None = None,
) -> dict[str, Any]:
    """Tek bir anlık ölçüm: tüm döngüleri değerlendirir ve sıralar.

    Args:
        payload: Ticker yanıtı.
        fee: İşlem başına komisyon. ``None`` ise config'ten.
        start: Döngülerin başlayacağı varlık (örn. ``"TL"``).

    Returns:
        ``evren``, ``dongu_sayisi``, ``sonuclar`` (net getiriye göre azalan) ve
        ``net_pozitif`` (kârlı döngü listesi).
    """
    f = CONFIG.backtest.commission_rate if fee is None else fee
    universe = describe_universe(payload)
    markets = build_markets(payload)
    cycles = find_cycles(markets, start=start)

    results = [r for r in (evaluate_cycle(cy, markets, f) for cy in cycles) if r]
    results.sort(key=lambda r: r["net_getiri"], reverse=True)
    return {
        "evren": universe,
        "komisyon": f,
        "dongu_sayisi": len(results),
        "sonuclar": results,
        "net_pozitif": [r for r in results if r["kar_var"]],
    }


# --------------------------------------------------------------------------- #
# Raporlama
# --------------------------------------------------------------------------- #


def print_universe(universe: dict[str, Any]) -> None:
    """Parite evrenini basar (üçgen mümkün mü, önce bu görünmeli)."""
    print("=" * 78)
    print("PARIBU PARITE EVRENI")
    print("=" * 78)
    print(f"  Toplam sembol     : {len(universe['tum_semboller'])}")
    print(f"  Kullanilabilir    : {len(universe['gecerli'])} (ask/bid dolu)")
    if universe["kullanilamaz"]:
        print(f"  Kullanilamaz      : {len(universe['kullanilamaz'])} -> "
              f"{', '.join(universe['kullanilamaz'][:8])}")
    print(f"  Para birimleri    : {', '.join(universe['para_birimleri'])}")
    print(f"\n  COIN-COIN pariteler ({len(universe['coin_coin'])}):")
    if universe["coin_coin"]:
        for s in universe["coin_coin"]:
            print(f"    {s}")
    else:
        print("    YOK")
        print("\n  >>> UCGEN ARBITRAJ YAPISAL OLARAK IMKANSIZ.")
        print("      Tum pariteler TL'ye karsi; TL -> X -> Y -> TL dongusunun")
        print("      orta bacagi (X_Y paritesi) borsada mevcut degil.")


def print_measurement(m: dict[str, Any], top: int = 10) -> None:
    """Ölçüm sonucunu basar; fırsat yoksa bunu açıkça söyler."""
    print("\n" + "=" * 78)
    print(f"UCGEN ARBITRAJ OLCUMU  |  komisyon {m['komisyon'] * 100:.3f}%/islem "
          f"(3 islem = {m['komisyon'] * 300:.2f}%)")
    print("=" * 78)

    if m["dongu_sayisi"] == 0:
        print("  Degerlendirilebilir dongu YOK.")
        return

    print(f"  {m['dongu_sayisi']} dongu degerlendirildi. En iyi {min(top, m['dongu_sayisi'])}:\n")
    print(f"  {'dongu':<28} {'brut %':>10} {'net %':>10}  kar")
    print("  " + "-" * 60)
    for r in m["sonuclar"][:top]:
        flag = "EVET" if r["kar_var"] else "hayir"
        print(f"  {r['dongu']:<28} {r['brut_getiri'] * 100:>+10.4f} "
              f"{r['net_getiri'] * 100:>+10.4f}  {flag}")

    n_pos = len(m["net_pozitif"])
    print()
    if n_pos == 0:
        best = m["sonuclar"][0]
        print("  >>> NET POZITIF DONGU YOK. Su anda uygulanabilir firsat BULUNMUYOR.")
        print(f"      En iyi dongu bile komisyon sonrasi {best['net_getiri'] * 100:+.4f}% "
              "veriyor (zarar).")
        if best["brut_getiri"] > 0:
            print(f"      Brut {best['brut_getiri'] * 100:+.4f}% pozitif olsa bile 3 islemlik "
                  f"{m['komisyon'] * 300:.2f}% komisyon onu yutuyor.")
        else:
            print(f"      Brut getiri ({best['brut_getiri'] * 100:+.4f}%) komisyon HARIC bile "
                  "negatif:")
            print("      fiyatlar tutarli, aradaki fark tamamen spread'ten geliyor.")
    else:
        print(f"  >>> {n_pos} dongu net POZITIF gorunuyor:")
        for r in m["net_pozitif"]:
            print(f"      {r['dongu']}: net {r['net_getiri'] * 100:+.4f}%")
        print("      UYARI: bu anlik bir olcumdur. Emirler dolana kadar fiyat")
        print("      degisebilir; ayrica defter derinligi kontrol EDILMEDI —")
        print("      ilan edilen ask/bid yalnizca en ust seviyedir.")


def print_legs(result: dict[str, Any]) -> None:
    """Bir döngünün bacak bacak detayını basar."""
    print(f"\n  {result['dongu']}  (net {result['net_getiri'] * 100:+.4f}%)")
    for i, leg in enumerate(result["bacaklar"], 1):
        print(f"    {i}. {leg['yon']:<3} {leg['parite']:<12} @ {leg['fiyat']:>18,.8f}"
              f"  ({leg['fiyat_tipi']})  {leg['girdi']:.8f} -> {leg['cikti']:.8f}")


# --------------------------------------------------------------------------- #
# İzleme (watch)
# --------------------------------------------------------------------------- #


def watch(
    duration: float,
    interval: float,
    fee: float | None = None,
    start: str | None = None,
    getter: Callable[[], dict[str, Any]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Belirli süre boyunca tekrar tekrar ölçer ve fırsat sayar.

    Üçgen arbitraj fırsatları ANLIKTIR: tek bir ölçüm yanıltıcıdır. Bu mod
    süre boyunca örnekleyip "kaç kez ve hangi büyüklükte" fırsat çıktığını
    sayar.

    Args:
        duration: Toplam izleme süresi (saniye).
        interval: Ölçümler arası bekleme (saniye).
        fee: İşlem başına komisyon.
        start: Döngü başlangıç varlığı.
        getter: Enjekte edilebilir veri sağlayıcı (test için).
        sleeper: Enjekte edilebilir bekleme (test için).
        clock: Enjekte edilebilir saat (test için).

    Returns:
        ``ornek``, ``hata``, ``firsatli_ornek``, ``firsat_orani``,
        ``en_iyi_net``, ``dongu_bazinda`` özetleri.
    """
    t0 = clock()
    samples = 0
    errors = 0
    with_opportunity = 0
    best_net = float("-inf")
    per_cycle: dict[str, list[float]] = {}

    while clock() - t0 < duration:
        try:
            payload = fetch_ticker(getter)
            m = measure(payload, fee=fee, start=start)
        except Exception as e:  # noqa: BLE001 - izleme dongusu dusmemeli
            errors += 1
            _log(f"olcum hatasi: {type(e).__name__}: {e}")
            sleeper(interval)
            continue

        samples += 1
        if m["sonuclar"]:
            top = m["sonuclar"][0]
            best_net = max(best_net, top["net_getiri"])
            for r in m["sonuclar"]:
                per_cycle.setdefault(r["dongu"], []).append(r["net_getiri"])
        if m["net_pozitif"]:
            with_opportunity += 1
            for r in m["net_pozitif"]:
                _log(f"FIRSAT: {r['dongu']} net {r['net_getiri'] * 100:+.4f}%")

        sleeper(interval)

    summary: list[dict[str, Any]] = []
    for cyc, vals in per_cycle.items():
        summary.append({
            "dongu": cyc,
            "ornek": len(vals),
            "ort_net": statistics.fmean(vals),
            "en_iyi_net": max(vals),
            "pozitif_sayisi": sum(1 for v in vals if v > 0),
        })
    summary.sort(key=lambda r: r["en_iyi_net"], reverse=True)

    return {
        "ornek": samples,
        "hata": errors,
        "firsatli_ornek": with_opportunity,
        "firsat_orani": with_opportunity / samples if samples else 0.0,
        "en_iyi_net": best_net if best_net > float("-inf") else float("nan"),
        "dongu_bazinda": summary,
    }


def print_watch_summary(w: dict[str, Any], top: int = 10) -> None:
    """İzleme özetini basar."""
    print("\n" + "=" * 78)
    print("IZLEME OZETI")
    print("=" * 78)
    print(f"  Olcum sayisi      : {w['ornek']}  (hata: {w['hata']})")
    print(f"  Firsatli olcum    : {w['firsatli_ornek']}  ({w['firsat_orani']:.1%})")
    if w["ornek"] == 0:
        print("\n  Hic basarili olcum yapilamadi.")
        return
    print(f"  Gorulen en iyi net: {w['en_iyi_net'] * 100:+.4f}%")
    if w["dongu_bazinda"]:
        print(f"\n  {'dongu':<28} {'ornek':>7} {'ort net %':>12} {'en iyi %':>11} {'poz':>6}")
        print("  " + "-" * 68)
        for r in w["dongu_bazinda"][:top]:
            print(f"  {r['dongu']:<28} {r['ornek']:>7} {r['ort_net'] * 100:>+12.4f} "
                  f"{r['en_iyi_net'] * 100:>+11.4f} {r['pozitif_sayisi']:>6}")
    if w["firsatli_ornek"] == 0:
        print("\n  >>> IZLEME BOYUNCA HIC NET POZITIF FIRSAT CIKMADI.")
        print("      Bu beklenen sonuctur: komisyon (3 x %0.2) cogu tutarsizligi yutar.")


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
        description="Paribu ici ucgen arbitraj OLCUM araci (islem yapmaz, anahtarsiz).",
    )
    ap.add_argument("--fee", type=float, default=None,
                    help=f"Islem basina komisyon (varsayilan {CONFIG.backtest.commission_rate}).")
    ap.add_argument("--start", type=str, default=None,
                    help="Dongulerin baslayacagi varlik (or. TL).")
    ap.add_argument("--watch", type=float, default=None,
                    help="Izleme suresi (saniye). Verilirse tekrarli olcum yapilir.")
    ap.add_argument("--interval", type=float, default=5.0, help="Olcumler arasi bekleme (sn).")
    ap.add_argument("--list-only", action="store_true", help="Sadece parite evrenini raporla.")
    ap.add_argument("--top", type=int, default=10, help="Kac dongu gosterilsin.")
    ap.add_argument("--legs", action="store_true", help="En iyi dongunun bacak detayini goster.")
    args = ap.parse_args(argv)

    print("NOT: Bu arac YALNIZCA OLCER. Emir gondermez, anahtar kullanmaz.\n")

    try:
        payload = fetch_ticker()
    except Exception as e:  # noqa: BLE001
        print(f"Ticker alinamadi: {type(e).__name__}: {e}")
        print("\nOlasi nedenler: ag erisimi kapali, rate limit, veya uc degismis.")
        print("Uc dogrulamasi icin: python3 research/tools/probe_paribu.py")
        return 1

    universe = describe_universe(payload)
    print_universe(universe)
    if args.list_only:
        return 0
    if not universe["ucgen_mumkun"]:
        print("\nUcgen kurulamadigi icin olcum yapilmadi.")
        return 1

    m = measure(payload, fee=args.fee, start=args.start)
    print_measurement(m, top=args.top)
    if args.legs and m["sonuclar"]:
        print_legs(m["sonuclar"][0])

    if args.watch:
        print(f"\n{args.watch:.0f} saniye boyunca {args.interval:.0f}s araliklarla izleniyor...")
        w = watch(args.watch, args.interval, fee=args.fee, start=args.start)
        print_watch_summary(w, top=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
