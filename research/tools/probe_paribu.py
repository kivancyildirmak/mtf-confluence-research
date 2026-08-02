#!/usr/bin/env python3
"""Paribu public (anahtarsız) market-data uçlarını KEŞFETME aracı.

NEDEN VAR? Bu betiği yazan ortamın çıkış (egress) politikası ``paribu.com``
alan adlarını kapatıyordu; resmî doküman okunamadı ve canlı istek atılamadı.
Doğrulanmamış endpoint yollarını "doğrulanmış" gibi koda gömmek yerine, işi
tersine çeviriyoruz: bu betik SİZİN makinenizde çalışır, adayları tek tek
dener ve GERÇEKTE ne cevap verdiğini raporlar.

Çıktısı (``paribu_probe.json``) doğrudan `research/data.py` içindeki
`fetch_paribu_ohlcv` / `fetch_orderbook` fonksiyonlarını yazmak için gereken
olgusal temeli sağlar.

ÖNEMLİ: Aşağıdaki ``CANDIDATES`` listesi **doğrulanmış gerçekler değil, test
edilecek adaylardır**. Betik hiçbir şeyi varsaymaz; yalnızca HTTP durum kodunu,
içerik tipini ve yanıtın yapısını raporlar. 200 dönmeyen aday sadece "bu aday
yanlış" demektir.

Bağımlılık YOK (yalnızca standart kütüphane). Anahtar/imza YOK — tamamen public.

Kullanım:

    python3 research/tools/probe_paribu.py
    python3 research/tools/probe_paribu.py --dump-docs   # dokümanı da indir
    python3 research/tools/probe_paribu.py --out /tmp/paribu_probe.json
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

#: Denenecek ADAY uçlar. Bunlar doğrulanmış yollar DEĞİL; hangisinin gerçekten
#: var olduğunu betik ölçecek. Yeni aday eklemek serbesttir.
CANDIDATES: list[tuple[str, str]] = [
    ("ticker", "https://www.paribu.com/ticker"),
    ("ticker", "https://v1.paribu.com/ticker"),
    ("ticker", "https://api.paribu.com/ticker"),
    ("ticker", "https://api.paribu.com/v1/ticker"),
    ("ticker", "https://www.paribu.com/api/ticker"),
    ("orderbook", "https://www.paribu.com/orderbook?symbol=BTC_TL"),
    ("orderbook", "https://v1.paribu.com/orderbook?symbol=BTC_TL"),
    ("orderbook", "https://api.paribu.com/v1/orderbook?symbol=BTC_TL"),
    ("orderbook", "https://api.paribu.com/orderbook/BTC_TL"),
    ("trades", "https://www.paribu.com/trades?symbol=BTC_TL"),
    ("trades", "https://api.paribu.com/v1/trades?symbol=BTC_TL"),
    ("candles", "https://api.paribu.com/v1/candles?symbol=BTC_TL&interval=1m"),
    ("candles", "https://api.paribu.com/v1/klines?symbol=BTC_TL&interval=1m"),
    ("candles", "https://www.paribu.com/chart/BTC_TL/1"),
    ("docs", "https://docs.paribu.com/api"),
]

USER_AGENT = "paribu-research-probe/1.0 (+public market data discovery)"
TIMEOUT = 20.0
SLEEP_BETWEEN = 1.0  # rate-limit nezaketi


def describe(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Bir JSON değerinin ŞEKLİNİ özetler (veriyi değil, yapıyı).

    Yanıtın tamamını basmak yerine alan adlarını ve tiplerini çıkarır; böylece
    çıktı hem kısa kalır hem de sütun eşlemesi için yeterli bilgi verir.

    Args:
        value: İncelenecek JSON değeri.
        depth: Mevcut derinlik.
        max_depth: Azami derinlik.

    Returns:
        Yapı özeti.
    """
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        items = list(value.items())[:12]
        return {k: describe(v, depth + 1, max_depth) for k, v in items}
    if isinstance(value, list):
        return [describe(value[0], depth + 1, max_depth), f"... ({len(value)} eleman)"] if value else []
    return type(value).__name__


def probe(url: str) -> dict[str, Any]:
    """Tek bir URL'yi dener ve sonucu raporlar.

    Args:
        url: Denenecek adres.

    Returns:
        Durum kodu, içerik tipi, yanıt yapısı veya hata bilgisi.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,*/*"})
    ctx = ssl.create_default_context()
    out: dict[str, Any] = {"url": url}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
            raw = resp.read(400_000)
            out["http_status"] = resp.status
            out["content_type"] = resp.headers.get("Content-Type", "")
            out["bytes"] = len(raw)
            text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(text)
                out["json"] = True
                out["shape"] = describe(data)
                out["ornek"] = text[:600]
            except json.JSONDecodeError:
                out["json"] = False
                out["ornek"] = text[:600]
    except urllib.error.HTTPError as e:
        out["http_status"] = e.code
        out["hata"] = f"HTTPError: {e.reason}"
        try:
            out["ornek"] = e.read(600).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        out["http_status"] = None
        out["hata"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    """Tüm adayları dener, özet basar ve JSON rapor yazar.

    Returns:
        Çıkış kodu (çalışan uç bulunduysa 0).
    """
    ap = argparse.ArgumentParser(description="Paribu public market-data uç keşfi (anahtarsız).")
    ap.add_argument("--out", default="paribu_probe.json", help="JSON rapor yolu.")
    ap.add_argument("--dump-docs", action="store_true", help="docs.paribu.com/api sayfasını da kaydet.")
    ap.add_argument("--symbol", default="BTC_TL", help="Denenecek sembol.")
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    print(f"{len(CANDIDATES)} aday uç deneniyor (anahtarsız, public)...\n")

    for kind, url in CANDIDATES:
        url = url.replace("BTC_TL", args.symbol)
        r = probe(url)
        r["kind"] = kind
        results.append(r)
        status = r.get("http_status")
        mark = "OK  " if status == 200 else "----"
        detail = r.get("hata", r.get("content_type", ""))
        print(f"  [{mark}] {status!s:>5}  {kind:10s} {url}")
        if status == 200 and r.get("json"):
            print(f"          yapi: {json.dumps(r['shape'], ensure_ascii=False)[:300]}")
        elif detail:
            print(f"          {str(detail)[:160]}")
        time.sleep(SLEEP_BETWEEN)

    if args.dump_docs:
        print("\nDoküman sayfası indiriliyor...")
        d = probe("https://docs.paribu.com/api")
        if d.get("http_status") == 200:
            p = Path("paribu_docs.html")
            p.write_text(d.get("ornek", ""), encoding="utf-8")
            print(f"  kaydedildi: {p} (ilk 600 karakter; tamamı icin tarayicidan kaydedin)")
        else:
            print(f"  alinamadi: {d.get('hata', d.get('http_status'))}")

    working = [r for r in results if r.get("http_status") == 200]
    Path(args.out).write_text(
        json.dumps({"results": results, "calisan_sayisi": len(working)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{len(working)}/{len(results)} aday 200 dondu. Rapor: {args.out}")
    if working:
        print("\nCALISAN UCLAR:")
        for r in working:
            print(f"  {r['kind']:10s} {r['url']}")
        print("\nBu raporu paylasin; dogrulanmis yollarla data.py yazilabilir.")
    else:
        print(
            "\nHicbir aday 200 dondurmedi. Bu, uclarin var olmadigi anlamina GELMEZ;\n"
            "aday listesi yanlis olabilir. Bu durumda docs.paribu.com/api sayfasini\n"
            "tarayicidan acip market-data bolumunu paylasin."
        )
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
