#!/usr/bin/env python3
"""ÇÖKÜŞ DENEYİ: grid stratejisini sert düşüş dönemlerinde stres testine sokar.

**AMAÇ PARA KAZANDIĞINI GÖSTERMEK DEĞİL, EN KÖTÜ SENARYODA NE KAYBETTİĞİNİ
ÖLÇMEKTİR.** Grid'i yalnızca fiyatın bantta salındığı sakin dönemlerde görmek
yanıltıcıdır; asıl risk fiyat aralığın altına kırıldığında ortaya çıkar.

**LOOK-AHEAD KARŞITI TASARIM (deneyin en kritik parçası).** Grid aralığı
dönemin tamamına bakarak seçilmez — bu, gerçekte bilinemeyecek bilgiyi
kullanmak olurdu ve grid'i asla kırılmayacak bir bantta gösterip çöküş riskini
tamamen görünmez kılardı. Bunun yerine aralık **yalnızca ilk 7 günün** fiyat
aralığından türetilir; o 7 gün işlem yapılmadan gözlenir ve işlem sonra başlar.
Böylece "grid kuruldu, sonra fiyat çöktü" senaryosu gerçekçi simüle edilir.

**Test edilen varyantlar:** static, trailing, static+stop-loss,
trailing+stop-loss. Stop-loss, fiyat alt sınırın X% altına inince tüm
pozisyonu satıp durur (düşen bıçağı yakalamayı bırakır).

Kullanım::

    # Gercek veri (once indirin)
    python -m research.tools.crash_experiment --download
    python -m research.tools.crash_experiment

    # Veri yoksa: sentetik stres senaryolari (GERCEK PIYASA DEGIL)
    python -m research.tools.crash_experiment --synthetic
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CONFIG, GridConfig
from ..data import load_ohlcv, resample_ohlcv, validate_ohlcv
from ..grid_backtest import buy_and_hold, run_grid_backtest


@dataclass(frozen=True)
class CrashPeriod:
    """Tarihsel bir çöküş penceresi.

    Attributes:
        tag: Kısa etiket (dosya adında kullanılır).
        name: Okunabilir ad.
        start: Başlangıç tarihi (UTC, ``YYYY-MM-DD``).
        end: Bitiş tarihi.
        note: Olayın kısa açıklaması.
        synthetic_drawdown: Sentetik karşılığı üretilirken kullanılacak düşüş.
    """

    tag: str
    name: str
    start: str
    end: str
    note: str
    synthetic_drawdown: float


#: Hedeflenen çöküş pencereleri. Düşüş oranları yaklaşık BTC hareketleridir;
#: gerçek veriyle çalışıldığında ölçülen değer raporda ayrıca basılır.
CRASH_PERIODS: tuple[CrashPeriod, ...] = (
    CrashPeriod("china2021", "2021 Mayis - Cin yasagi", "2021-05-01", "2021-06-30",
                "Cin madencilik/islem yasagi; BTC ~58k -> ~30k", 0.50),
    CrashPeriod("luna2022", "2022 Mayis - LUNA/UST cokusu", "2022-05-01", "2022-06-15",
                "UST peg kaybi, LUNA sifirlandi; BTC ~39k -> ~26k", 0.35),
    CrashPeriod("ftx2022", "2022 Kasim - FTX cokusu", "2022-11-01", "2022-12-15",
                "FTX iflasi, bulasma etkisi; BTC ~21k -> ~15.5k", 0.28),
)

#: Stop-loss varyantında kullanılan eşik: alt sınırın %5 altı.
DEFAULT_STOP_PCT = 0.05


def _log(msg: str) -> None:
    """İlerleme çıktısı."""
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #


def download_periods(out_dir: str | Path | None = None) -> dict[str, Path]:
    """Çöküş pencerelerini Binance'ten indirir.

    Args:
        out_dir: Çıktı klasörü.

    Returns:
        ``tag -> dosya yolu`` eşlemesi (yalnızca başarılı inenler).
    """
    from .fetch_binance import fetch_history

    saved: dict[str, Path] = {}
    for p in CRASH_PERIODS:
        _log(f"\n--- {p.name} ({p.start} -> {p.end}) ---")
        try:
            bars, meta = fetch_history(start=p.start, end=p.end, tag=p.tag, out_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            _log(f"  INDIRILEMEDI: {type(e).__name__}: {e}")
            continue
        if bars.empty or not meta.get("dosya"):
            _log(f"  Veri gelmedi (sembol={meta.get('secilen_sembol')}).")
            continue
        drop = float(bars["close"].min() / bars["close"].iloc[0] - 1.0)
        _log(f"  {len(bars):,} bar | sembol={meta['secilen_sembol']} | "
             f"olculen dip dususu {drop:.1%}")
        saved[p.tag] = Path(meta["dosya"])
    return saved


def synthetic_crash(
    period: CrashPeriod,
    days: int = 50,
    calm_days: int = 12,
    crash_days: int = 6,
    bar_minutes: int = 60,
    start_price: float = 100_000.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Bir çöküş senaryosunun SENTETİK karşılığını üretir.

    **BU GERÇEK PİYASA VERİSİ DEĞİLDİR.** Yalnızca motorun çöküşte nasıl
    davrandığını görmek içindir. Üç evreli kurgu, gerçek çöküşlerin şeklini
    taklit eder:

    1. **Sakin evre** — dar bantta salınım. Grid burada kurulur ve
       gerçekleşmiş kâr biriktirir.
    2. **Çöküş evresi** — ``synthetic_drawdown`` kadar sert düşüş.
    3. **Sonrası** — düşük seviyede dalgalı seyir, kısmi toparlanma.

    Args:
        period: Taklit edilecek dönem (düşüş oranı buradan alınır).
        days: Toplam gün.
        calm_days: Sakin evre uzunluğu.
        crash_days: Çöküş evresi uzunluğu.
        bar_minutes: Bar periyodu.
        start_price: Başlangıç fiyatı.
        seed: Rastgelelik tohumu.

    Returns:
        Doğrulanmış OHLCV çerçevesi.
    """
    rng = np.random.default_rng(seed)
    per_day = int(24 * 60 / bar_minutes)
    n_calm, n_crash = calm_days * per_day, crash_days * per_day
    n_after = max(days - calm_days - crash_days, 1) * per_day
    n = n_calm + n_crash + n_after

    bar_vol = 0.6 / np.sqrt(365 * per_day)  # ~%60 yillik vol
    drift = np.zeros(n)
    # Cokus evresinde negatif surukleme: toplam log dususu hedefe esitle.
    total_log_drop = np.log(1.0 - period.synthetic_drawdown)
    drift[n_calm : n_calm + n_crash] = total_log_drop / n_crash
    # Sonrasinda hafif toparlanma (dususun ~%20'si geri gelir).
    drift[n_calm + n_crash :] = -0.20 * total_log_drop / n_after

    shocks = rng.standard_normal(n) * bar_vol
    shocks[n_calm : n_calm + n_crash] *= 2.5  # cokuste vol patlamasi
    log_close = np.log(start_price) + np.cumsum(drift + shocks)
    close = np.exp(log_close)

    open_ = np.concatenate([[start_price], close[:-1]])
    span = np.abs(shocks) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    low = np.maximum(low, close * 0.5)

    idx = pd.date_range("2022-01-01", periods=n, freq=f"{bar_minutes}min", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.abs(rng.standard_normal(n)) * 10 + 1},
        index=idx,
    )
    df.index.name = "timestamp"
    return validate_ohlcv(df)


def load_period(path: str | Path, resample: str = "1h") -> pd.DataFrame:
    """Dönem verisini yükler ve 1 saatlik bara indirir.

    Args:
        path: Parquet/CSV yolu.
        resample: Hedef bar.

    Returns:
        OHLCV çerçevesi.
    """
    df = load_ohlcv(path)
    return resample_ohlcv(df, resample) if resample else df


# --------------------------------------------------------------------------- #
# Varyantlar
# --------------------------------------------------------------------------- #


def build_variants(
    base: GridConfig | None = None,
    stop_pct: float = DEFAULT_STOP_PCT,
) -> dict[str, GridConfig]:
    """Karşılaştırılacak grid varyantlarını üretir.

    Args:
        base: Temel konfigürasyon.
        stop_pct: Stop-loss eşiği (alt sınırın altında, oran).

    Returns:
        ``ad -> GridConfig`` eşlemesi.
    """
    b = copy.deepcopy(base or CONFIG.grid)
    # Aralik ISINMA penceresinden turetilecek: acik sinir verilmez.
    b.lower_price = None
    b.upper_price = None

    variants: dict[str, GridConfig] = {}
    for mode in ("static", "trailing"):
        v = copy.deepcopy(b)
        v.mode = mode
        v.stop_loss_pct = None
        variants[mode] = v

        vs = copy.deepcopy(b)
        vs.mode = mode
        vs.stop_loss_pct = stop_pct
        variants[f"{mode}+stop"] = vs
    return variants


def run_period(
    name: str,
    df: pd.DataFrame,
    base: GridConfig | None = None,
    stop_pct: float = DEFAULT_STOP_PCT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tek bir dönemde tüm varyantları çalıştırır.

    Args:
        name: Dönem adı.
        df: OHLCV verisi.
        base: Temel grid konfigürasyonu.
        stop_pct: Stop-loss eşiği.

    Returns:
        ``(sonuc_tablosu, donem_bilgisi)``.
    """
    variants = build_variants(base, stop_pct)
    rows: list[dict[str, Any]] = []
    info: dict[str, Any] = {"donem": name, "bar": len(df)}

    for vname, vcfg in variants.items():
        try:
            res = run_grid_backtest(df, vcfg)
        except ValueError as e:
            _log(f"  {vname}: calistirilamadi ({e})")
            continue
        m = res.metrics
        bh = buy_and_hold(df, vcfg)
        if "aralik" not in info:
            info["aralik"] = (res.config["alt_sinir"], res.config["ust_sinir"])
            info["isinma"] = m.get("isinma", {})
            info["al_tut"] = bh["getiri"]
            info["al_tut_dd"] = bh["max_drawdown"]
        rows.append({
            "donem": name,
            "varyant": vname,
            "gerceklesmis": m["gerceklesmis_kar_oran"],
            "gerceklesmemis": m["gerceklesmemis_kz_oran"],
            "TOPLAM": m["toplam_getiri"],
            "max_dd": m["max_drawdown"],
            "aralik_kirildi": m["aralik_kirildi"],
            "aralik_disi": m["aralik_disi_sure_orani"],
            "elde_kalan_deger": m["elde_kalan_deger"],
            "stop": m["stop_tetiklendi"],
            "cift": m["tamamlanan_cift"],
            "al_tut": bh["getiri"],
            "al_tut_farki": m["toplam_getiri"] - bh["getiri"],
        })
    return pd.DataFrame(rows), info


def print_period_report(name: str, df: pd.DataFrame, table: pd.DataFrame,
                        info: dict[str, Any]) -> None:
    """Bir dönemin tam dürüst raporunu basar."""
    print("\n" + "=" * 100)
    print(f"DONEM: {name}")
    print("=" * 100)
    px = df["close"]
    dip = float(px.min() / px.iloc[0] - 1.0)
    son = float(px.iloc[-1] / px.iloc[0] - 1.0)
    print(f"  Bar            : {len(df):,} ({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    print(f"  Fiyat          : {px.iloc[0]:,.0f} -> {px.iloc[-1]:,.0f}  "
          f"(dip {dip:.1%}, donem sonu {son:.1%})")
    w = info.get("isinma", {})
    if w.get("kullanildi"):
        print(f"  Isinma (islemsiz): {w['gun']:.0f} gun, {w['bar']:,} bar  "
              f"[{w['baslangic']:%Y-%m-%d} -> {w['bitis']:%Y-%m-%d}]")
    if "aralik" in info:
        lo, hi = info["aralik"]
        print(f"  Grid araligi   : {lo:,.0f} - {hi:,.0f}  (yalnizca isinmadan turetildi)")
        print(f"  Fiyat/alt sinir: donem dibi alt sinirin {px.min() / lo - 1:.1%} "
              f"{'ALTINDA' if px.min() < lo else 'ustunde'}")
    print(f"  Al-ve-tut      : {info.get('al_tut', float('nan')):+.2%}  "
          f"(maxDD {info.get('al_tut_dd', float('nan')):.2%})")

    if table.empty:
        print("\n  Hicbir varyant calistirilamadi.")
        return
    show = table.drop(columns=["donem", "al_tut"]).copy()
    for col in ("gerceklesmis", "gerceklesmemis", "TOPLAM", "max_dd",
                "aralik_disi", "al_tut_farki"):
        show[col] = show[col].map(lambda v: f"{v:+.2%}")
    show["elde_kalan_deger"] = show["elde_kalan_deger"].map(lambda v: f"{v:,.0f}")
    print()
    print(show.to_string(index=False))


def summary_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Dönem × varyant özet tablosu (toplam getiri ve max drawdown yan yana).

    Args:
        all_rows: Tüm dönemlerin birleştirilmiş sonuçları.

    Returns:
        Çift indeksli özet tablo.
    """
    if all_rows.empty:
        return pd.DataFrame()
    ret = all_rows.pivot(index="donem", columns="varyant", values="TOPLAM")
    dd = all_rows.pivot(index="donem", columns="varyant", values="max_dd")
    out = pd.concat({"toplam_getiri": ret, "max_drawdown": dd}, axis=1)
    return out.round(4)


def stop_loss_benefit(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Stop-loss'un zararı ne kadar azalttığını ölçer.

    Args:
        all_rows: Tüm dönemlerin sonuçları.

    Returns:
        Dönem × mod bazında stop'lu ve stop'suz karşılaştırması.
    """
    rows: list[dict[str, Any]] = []
    for (period, mode), _ in all_rows.groupby(
        ["donem", all_rows["varyant"].str.replace("+stop", "", regex=False)]
    ):
        sub = all_rows[(all_rows["donem"] == period)]
        plain = sub[sub["varyant"] == mode]
        stopped = sub[sub["varyant"] == f"{mode}+stop"]
        if plain.empty or stopped.empty:
            continue
        p, s = plain.iloc[0], stopped.iloc[0]
        rows.append({
            "donem": period,
            "mod": mode,
            "stopsuz": p["TOPLAM"],
            "stoplu": s["TOPLAM"],
            "fark": s["TOPLAM"] - p["TOPLAM"],
            "zarar_azalmasi": (
                1.0 - s["TOPLAM"] / p["TOPLAM"] if p["TOPLAM"] < 0 else float("nan")
            ),
            "stop_tetiklendi": bool(s["stop"]),
            "stopsuz_dd": p["max_dd"],
            "stoplu_dd": s["max_dd"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


SYNTHETIC_BANNER = """
################################################################################
DIKKAT: SENTETIK STRES SENARYOLARI KULLANILIYOR — GERCEK PIYASA VERISI DEGIL.
################################################################################
Binance indirmesi bu ortamda yapilamadi (egress politikasi). Asagidaki seriler,
gercek olaylarin BUYUKLUGUNU taklit eden kurgusal fiyat yollaridir. Motorun
cokuste nasil davrandigini gosterirler; "2022 Mayis'ta sunu kazanirdiniz" gibi
bir sey SOYLEMEZLER. Gercek sonuc icin once veriyi indirin:
    python -m research.tools.crash_experiment --download
################################################################################
"""


def main(argv: list[str] | None = None) -> int:
    """Komut satırı giriş noktası.

    Args:
        argv: Argümanlar (test için).

    Returns:
        Çıkış kodu.
    """
    ap = argparse.ArgumentParser(
        description="Grid stratejisinin cokus donemlerinde stres testi.",
    )
    ap.add_argument("--download", action="store_true", help="Once donemleri indir.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Veri yoksa sentetik stres senaryolari kullan.")
    ap.add_argument("--data-dir", type=str, default=None, help="Parquet klasoru.")
    ap.add_argument("--levels", type=int, default=None, help="Kademe sayisi.")
    ap.add_argument("--stop-pct", type=float, default=DEFAULT_STOP_PCT,
                    help="Stop-loss esigi (alt sinirin altinda, oran).")
    ap.add_argument("--warmup-days", type=float, default=7.0,
                    help="Aralik icin gozlem penceresi (gun).")
    args = ap.parse_args(argv)

    if args.download:
        _log("Cokus donemleri indiriliyor...")
        saved = download_periods(args.data_dir)
        _log(f"\n{len(saved)}/{len(CRASH_PERIODS)} donem indirildi.")
        if not saved:
            _log("Hicbiri inmedi; --synthetic ile motoru gorebilirsiniz.")
            return 1

    base = copy.deepcopy(CONFIG.grid)
    base.range_warmup_days = args.warmup_days
    if args.levels:
        base.n_levels = args.levels

    data_dir = Path(args.data_dir or CONFIG.collector.ohlcv_dir)
    periods: list[tuple[str, pd.DataFrame]] = []
    missing: list[str] = []

    for p in CRASH_PERIODS:
        hits = sorted(data_dir.glob(f"*{p.tag}*.parquet")) if data_dir.exists() else []
        if hits:
            periods.append((p.name, load_period(hits[0])))
        else:
            missing.append(p.name)

    if missing and not periods:
        if not args.synthetic:
            print(f"Veri bulunamadi ({data_dir}). Eksik donemler: {', '.join(missing)}")
            print("\nOnce indirin:  python -m research.tools.crash_experiment --download")
            print("Motoru gormek icin: --synthetic")
            return 1
        print(SYNTHETIC_BANNER)
        periods = []
        for i, p in enumerate(CRASH_PERIODS):
            gen = synthetic_crash(p, seed=i)
            # Etikette HEDEF degil, uretilen seride OLCULEN dip dususu yazilir:
            # cokus evresindeki vol patlamasi hedefin otesine gecebiliyor.
            olculen = float(gen["close"].min() / gen["close"].iloc[0] - 1.0)
            periods.append((f"[SENTETIK] {p.name} (dip {olculen:.0%})", gen))
    elif missing:
        print(f"UYARI: su donemler icin veri yok, atlandi: {', '.join(missing)}")

    all_rows: list[pd.DataFrame] = []
    for name, df in periods:
        table, info = run_period(name, df, base, args.stop_pct)
        print_period_report(name, df, table, info)
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if combined.empty:
        print("\nHic sonuc uretilemedi.")
        return 1

    print("\n" + "=" * 100)
    print("OZET: DONEM x VARYANT")
    print("=" * 100)
    print(summary_table(combined).to_string())

    print("\n" + "=" * 100)
    print("STOP-LOSS ZARARI NE KADAR AZALTTI?")
    print("=" * 100)
    benefit = stop_loss_benefit(combined)
    if not benefit.empty:
        print(benefit.round(4).to_string(index=False))

    print("\n--- YORUM ---")
    worst = combined.loc[combined["TOPLAM"].idxmin()]
    print(f"  En kotu sonuc: {worst['donem']} / {worst['varyant']} "
          f"-> {worst['TOPLAM']:+.2%} (maxDD {worst['max_dd']:.2%})")
    broke = combined["aralik_kirildi"].mean()
    print(f"  Aralik kirilma orani: {broke:.0%} (varyant-donem kombinasyonlarinda)")
    pos = (combined["TOPLAM"] > 0).mean()
    print(f"  Pozitif getiri orani: {pos:.0%}")

    # Cokuste trailing'in statikten farksiz olmasi BEKLENEN sonuctur; bunu
    # sessizce gecmek, trailing'i "test edilmis ama etkisiz" gostermek olurdu.
    same = []
    for period, grp in combined.groupby("donem"):
        st = grp[grp["varyant"] == "static"]["TOPLAM"]
        tr = grp[grp["varyant"] == "trailing"]["TOPLAM"]
        if not st.empty and not tr.empty and abs(float(st.iloc[0]) - float(tr.iloc[0])) < 1e-9:
            same.append(period)
    if same:
        print(f"  NOT: {len(same)}/{combined['donem'].nunique()} donemde trailing = static.")
        print("       Trailing yalnizca fiyat UST siniri astiginda kayar; cokuste")
        print("       fiyat hic yukari gitmedigi icin kaydirma tetiklenmez.")
        print("       Yani trailing cokus riskine karsi HICBIR koruma saglamaz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
