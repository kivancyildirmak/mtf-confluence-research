"""Grid (ızgara) stratejisi backtest motoru — en kötü senaryoyu gizlemeden.

Grid bir TAHMİN modeli değildir: fiyat bir kademe düşünce alır, bir kademe
çıkınca satar. ML hattından bağımsızdır ama **aynı maliyet ve dürüstlük
disiplinine** tabidir.

**GRID'İN GİZLİ RİSKİ.** Grid backtest'leri sektörde sistematik olarak
yanıltıcıdır çünkü genellikle yalnızca "gerçekleşmiş kâr" raporlanır:
tamamlanan al-sat çiftlerinin toplamı. Bu sayı neredeyse her zaman pozitiftir —
fiyat düştükçe grid almaya devam eder ve her yükselişte kâr kilitler. Gizlenen
şey, dönem sonunda **elde kalan coin**'dir: fiyat aralığın altına düştüğünde
strateji, giderek değer kaybeden bir envanterin üstünde oturur. Bu modül bu
yüzden şunları AYRI AYRI raporlar:

* ``gerceklesmis_kar`` — tamamlanan al-sat çiftleri (maliyet sonrası),
* ``gerceklesmemis_kz`` — dönem sonunda elde kalan envanterin kâr/zararı,
* ``toplam_kz`` — ikisinin toplamı (tek dürüst "kâr" ölçüsü),
* ``aralik_kirildi`` — fiyat alt sınırın ALTINA indi mi (strateji fiilen çöktü),
* ``aralik_disi_sure_orani`` — fiyatın aralık dışında geçirdiği zaman.

**İCRA VARSAYIMLARI (kötümser).**

1. Her işlemde komisyon + slippage uygulanır (varsayılan gidiş-dönüş ~%0.5).
2. Emirler kademe fiyatından dolar; slippage yön aleyhine eklenir.
3. Aynı barda alınan bir lot aynı barda SATILAMAZ — bar içi sıralama
   bilinmediği için sahte bar-içi scalping üretilmez.
4. Nakit biter ve yeni alım yapılamazsa alım ATLANIR ve sayılır. Düşüş
   trendinde grid'in gerçekten yaşadığı şey budur.
5. Seviyenin üstünde limit alım verilmez: bir kademede alım ancak fiyat oraya
   DÜŞTÜĞÜNDE tetiklenir (piyasa üstü alım fiziksel olarak anlamsızdır).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, GridConfig


@dataclass
class GridResult:
    """Grid backtest çıktılarının tamamı.

    Attributes:
        trades: İşlem bazında detay (alış/satış).
        pairs: Tamamlanmış al-sat çiftleri.
        equity: Bar bazında toplam varlık (nakit + envanter değeri).
        cash: Bar bazında nakit.
        inventory: Bar bazında elde tutulan coin miktarı.
        metrics: Özet metrikler (dürüstlük kalemleri dahil).
        levels: Kullanılan kademe fiyatları (son hâli).
        regimes: Alt dönem (rejim) analizi tablosu.
        config: Kullanılan varsayımlar.
    """

    trades: pd.DataFrame
    pairs: pd.DataFrame
    equity: pd.Series
    cash: pd.Series
    inventory: pd.Series
    metrics: dict[str, Any]
    levels: np.ndarray
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Kademeler
# --------------------------------------------------------------------------- #


def build_levels(
    lower: float,
    upper: float,
    n_levels: int,
    spacing: str = "geometric",
) -> np.ndarray:
    """Grid kademelerini üretir (artan sıralı).

    Args:
        lower: Alt sınır fiyatı.
        upper: Üst sınır fiyatı.
        n_levels: Kademe sayısı.
        spacing: ``"geometric"`` (eşit yüzde adım) veya ``"linear"``.

    Returns:
        ``n_levels`` uzunluğunda artan fiyat dizisi.

    Raises:
        ValueError: Sınırlar geçersizse veya kademe sayısı 2'den küçükse.

    Notes:
        Kripto için ``geometric`` daha doğrudur: her adım eşit YÜZDE hareket
        eder, dolayısıyla her al-sat çifti aynı yüzde kârı hedefler. Doğrusal
        aralıkta düşük fiyatlardaki adımlar yüzde olarak büyür.
    """
    if n_levels < 2:
        raise ValueError("n_levels en az 2 olmalı.")
    if not (0 < lower < upper):
        raise ValueError(f"Geçersiz aralık: lower={lower}, upper={upper}")
    if spacing == "geometric":
        return np.geomspace(lower, upper, n_levels)
    if spacing == "linear":
        return np.linspace(lower, upper, n_levels)
    raise ValueError(f"Bilinmeyen spacing: {spacing}")


def select_range_from_warmup(
    df: pd.DataFrame,
    warmup_days: float,
    pad_pct: float = 0.0,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """Aralığı YALNIZCA ilk ``warmup_days`` günün fiyatlarından belirler.

    **Neden şart?** Aralığı dönemin tamamına bakarak seçmek (örn. "dönemin en
    düşüğü alt sınır olsun") look-ahead'dir: gerçekte grid'i kurarken geleceği
    bilemezsiniz. O yanlış kurulum, grid'i asla kırılmayacak bir bant içinde
    gösterir ve çöküş riskini tamamen görünmez kılar.

    Bu fonksiyon ısınma penceresini **işlem dışı** tutar: o günler yalnızca
    gözlenir, aralık onlardan türetilir, işlem penceresi ondan SONRA başlar.

    Args:
        df: OHLCV verisi.
        warmup_days: Gözlem penceresi (gün).
        pad_pct: Aralığa eklenecek pay (üstten ve alttan, oran).

    Returns:
        ``(alt, ust, isinma_verisi, islem_verisi)``.

    Raises:
        ValueError: Isınma penceresi veriyi tümüyle tüketiyorsa veya
            pencerede geçerli fiyat yoksa.
    """
    if df.empty:
        raise ValueError("Boş veri.")
    cutoff = df.index[0] + pd.Timedelta(days=float(warmup_days))
    warmup = df.loc[df.index < cutoff]
    trade = df.loc[df.index >= cutoff]
    if warmup.empty:
        raise ValueError("Isınma penceresi boş.")
    if trade.empty:
        raise ValueError(
            f"Isınma ({warmup_days} gün) tüm veriyi tüketti; işlem dönemi kalmadı."
        )
    lo = float(warmup["low"].min())
    hi = float(warmup["high"].max())
    if not (0 < lo < hi):
        raise ValueError(f"Isınma penceresinden geçerli aralık çıkmadı: {lo}-{hi}")
    return lo * (1.0 - pad_pct), hi * (1.0 + pad_pct), warmup, trade


def grid_step_pct(levels: np.ndarray) -> float:
    """Ortalama kademe adımını yüzde olarak döndürür.

    Bu sayı hayati: adım, gidiş-dönüş maliyetten (%0.5) küçükse strateji
    matematiksel olarak kaybeder — her tamamlanan çift zarar yazar.

    Args:
        levels: Kademe fiyatları.

    Returns:
        Ortalama adım (oran).
    """
    if len(levels) < 2:
        return 0.0
    return float(np.mean(np.diff(levels) / levels[:-1]))


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #


def run_grid_backtest(
    df: pd.DataFrame,
    cfg: GridConfig | None = None,
) -> GridResult:
    """Grid stratejisini geçmiş veride çalıştırır.

    Args:
        df: OHLCV verisi (UTC indeksli, artan).
        cfg: Grid konfigürasyonu.

    Returns:
        :class:`GridResult`.

    Raises:
        ValueError: Veri boşsa veya şema bozuksa.

    Notes:
        SIZINTI: Karar yalnızca o barın fiyat aralığına bakar; gelecekteki
        hiçbir bar kullanılmaz. Bar içi emir sıralaması bilinmediği için aynı
        barda al-sat çifti tamamlanmasına izin verilmez (kötümser varsayım).
    """
    c = cfg or CONFIG.grid
    if df.empty:
        raise ValueError("Boş veri.")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Eksik sütun: {col}")

    # --- Aralık seçimi (look-ahead'siz) ---------------------------------- #
    warmup_info: dict[str, Any] = {"kullanildi": False}
    if c.lower_price is not None and c.upper_price is not None:
        lower, upper = c.lower_price, c.upper_price
    elif c.range_warmup_days and c.range_warmup_days > 0:
        lower, upper, warm, df = select_range_from_warmup(
            df, c.range_warmup_days, c.range_pad_pct
        )
        warmup_info = {
            "kullanildi": True,
            "gun": c.range_warmup_days,
            "bar": int(len(warm)),
            "baslangic": warm.index[0],
            "bitis": warm.index[-1],
        }
    else:
        p0 = float(df["open"].iloc[0])
        lower = p0 * (1.0 - c.auto_range_pct)
        upper = p0 * (1.0 + c.auto_range_pct)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)
    p0 = float(df["open"].iloc[0])
    levels = build_levels(lower, upper, c.n_levels, c.spacing)
    # Stop seviyesi başlangıç alt sınırına göre sabitlenir (grid kaysa bile).
    stop_price = lower * (1.0 - c.stop_loss_pct) if c.stop_loss_pct is not None else None

    slip = c.slippage_bps / 10_000.0
    quote_per_level = c.total_capital / c.n_levels

    # --- Durum ----------------------------------------------------------- #
    cash = c.total_capital
    held_qty = np.zeros(c.n_levels)      # kademe başına elde tutulan miktar
    held_cost = np.zeros(c.n_levels)     # kademe başına toplam maliyet (nakit çıkışı)
    # Satış hedefi ALIM ANINDA sabitlenir. Grid kaysa bile açık bir lotun çıkış
    # fiyatı geriye dönük DEĞİŞMEZ — gerçekte de verilmiş limit emri yerinde kalır.
    held_target = np.zeros(c.n_levels)
    held_time: list[Any] = [None] * c.n_levels

    trades: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    equity = np.zeros(n)
    cash_curve = np.zeros(n)
    inv_curve = np.zeros(n)

    realized = 0.0
    total_cost = 0.0
    skipped_buys = 0
    grid_shifts = 0
    below_lower = above_upper = 0
    range_broken = False
    prev_close = p0

    stopped = False
    stop_time: Any = None
    stop_realized = 0.0

    for i in range(n):
        ts = df.index[i]
        hi, lo, cl = high[i], low[i], close[i]

        # --- STOP-LOSS: alt sınırın X% altına inildiyse her şeyi sat, dur --- #
        # Kötümser icra: satış tam stop fiyatından değil, o barın DÜŞÜĞÜNDEN
        # yapılır — sert çöküşte stop emri genelde daha kötü dolar (gap).
        if stop_price is not None and not stopped and lo <= stop_price:
            fill = min(stop_price, lo) * (1.0 - slip)
            for k in range(len(held_qty)):
                if held_qty[k] <= 0:
                    continue
                qty = held_qty[k]
                proceeds = qty * fill
                commission = proceeds * c.commission_rate
                net = proceeds - commission
                pnl = net - held_cost[k]
                cash += net
                realized += pnl
                stop_realized += pnl
                total_cost += commission + proceeds * slip
                pairs.append({
                    "kademe": k, "alis_zamani": held_time[k], "satis_zamani": ts,
                    "alis_maliyeti": held_cost[k], "satis_neti": net, "kar": pnl,
                    "kar_orani": pnl / held_cost[k] if held_cost[k] else 0.0,
                })
                trades.append({
                    "zaman": ts, "yon": "STOP-SAT", "kademe": k, "fiyat": fill,
                    "miktar": qty, "tutar": proceeds, "komisyon": commission,
                })
                held_qty[k] = 0.0
                held_cost[k] = 0.0
                held_target[k] = 0.0
                held_time[k] = None
            stopped = True
            stop_time = ts

        if stopped:
            # Grid durdu: bundan sonra yalnızca nakit tutulur.
            cash_curve[i] = cash
            inv_curve[i] = 0.0
            equity[i] = cash
            prev_close = cl
            continue

        # --- Hareketli aralık: fiyat üst sınırı aşarsa grid bir kademe kayar - #
        # En alttaki kademe düşürülür, tepeye yeni bir kademe eklenir (klasik
        # trailing grid). Yalnızca en alt kademe BOŞSA kaydırılır; doluysa o lot
        # gridden düşerdi ve envanteri kaybederdik.
        if c.mode == "trailing":
            guard = 0
            while hi > levels[-1] and held_qty[0] <= 0 and guard < c.n_levels:
                step = levels[-1] / levels[-2]
                levels = np.append(levels[1:], levels[-1] * step)
                held_qty = np.append(held_qty[1:], 0.0)
                held_cost = np.append(held_cost[1:], 0.0)
                held_target = np.append(held_target[1:], 0.0)
                held_time = held_time[1:] + [None]
                grid_shifts += 1
                guard += 1

        # --- Aralık dışı takibi -------------------------------------------- #
        if lo < levels[0]:
            below_lower += 1
            range_broken = True
        if hi > levels[-1]:
            above_upper += 1

        bought_this_bar: set[int] = set()

        # --- ALIŞ: fiyat bir kademeye DÜŞTÜYSE ----------------------------- #
        # Yalnızca fiyatın altındaki (veya üstünden gelip değdiği) kademelerde.
        for k in range(c.n_levels - 1):  # en üst kademede alım yok (satış hedefi yok)
            lvl = levels[k]
            if held_qty[k] > 0:
                continue
            if not (lo <= lvl <= hi):
                continue
            if lvl > prev_close:
                continue  # piyasa üstü limit alım verilmez
            buy_price = lvl * (1.0 + slip)
            commission = quote_per_level * c.commission_rate
            need = quote_per_level + commission
            if cash < need:
                skipped_buys += 1
                continue
            qty = quote_per_level / buy_price
            cash -= need
            held_qty[k] = qty
            held_cost[k] = need
            held_target[k] = levels[k + 1]   # cikis fiyati simdi sabitlenir
            held_time[k] = ts
            total_cost += commission + quote_per_level * slip
            bought_this_bar.add(k)
            trades.append({
                "zaman": ts, "yon": "AL", "kademe": k, "fiyat": buy_price,
                "miktar": qty, "tutar": quote_per_level, "komisyon": commission,
            })

        # --- SATIŞ: elde lot varsa ve bir üst kademe değildiyse ------------ #
        for k in range(c.n_levels - 1):
            if held_qty[k] <= 0 or k in bought_this_bar:
                continue  # aynı barda al-sat YASAK (kötümser)
            target = held_target[k]
            if not (lo <= target <= hi):
                continue
            if target < prev_close:
                continue  # piyasa altı limit satış verilmez
            sell_price = target * (1.0 - slip)
            qty = held_qty[k]
            proceeds = qty * sell_price
            commission = proceeds * c.commission_rate
            net = proceeds - commission
            pnl = net - held_cost[k]
            cash += net
            realized += pnl
            total_cost += commission + proceeds * slip
            pairs.append({
                "kademe": k, "alis_zamani": held_time[k], "satis_zamani": ts,
                "alis_maliyeti": held_cost[k], "satis_neti": net, "kar": pnl,
                "kar_orani": pnl / held_cost[k] if held_cost[k] else 0.0,
            })
            trades.append({
                "zaman": ts, "yon": "SAT", "kademe": k, "fiyat": sell_price,
                "miktar": qty, "tutar": proceeds, "komisyon": commission,
            })
            held_qty[k] = 0.0
            held_cost[k] = 0.0
            held_target[k] = 0.0
            held_time[k] = None

        inv = float(held_qty.sum())
        cash_curve[i] = cash
        inv_curve[i] = inv
        equity[i] = cash + inv * cl
        prev_close = cl

    # --- Sonuçlar --------------------------------------------------------- #
    equity_s = pd.Series(equity, index=df.index, name="equity")
    final_price = float(close[-1])
    remaining_qty = float(held_qty.sum())
    remaining_cost = float(held_cost.sum())
    remaining_value = remaining_qty * final_price
    unrealized = remaining_value - remaining_cost

    metrics = _compute_metrics(
        equity_s, df, realized, unrealized, remaining_qty, remaining_value,
        remaining_cost, total_cost, len(pairs), len(trades), skipped_buys,
        grid_shifts, below_lower, above_upper, range_broken, levels, c, n,
    )
    metrics["stop_tetiklendi"] = bool(stopped)
    metrics["stop_zamani"] = stop_time
    metrics["stop_fiyati"] = stop_price
    metrics["stopta_gerceklesen_kz"] = stop_realized
    metrics["isinma"] = warmup_info

    return GridResult(
        trades=pd.DataFrame(trades),
        pairs=pd.DataFrame(pairs),
        equity=equity_s,
        cash=pd.Series(cash_curve, index=df.index, name="cash"),
        inventory=pd.Series(inv_curve, index=df.index, name="inventory"),
        metrics=metrics,
        levels=levels,
        regimes=regime_analysis(df, equity_s, c),
        config={
            "mode": c.mode, "n_levels": c.n_levels, "spacing": c.spacing,
            "alt_sinir": float(levels[0]), "ust_sinir": float(levels[-1]),
            "commission_rate": c.commission_rate, "slippage_bps": c.slippage_bps,
            "gidis_donus_maliyet": 2.0 * (c.commission_rate + c.slippage_bps / 10_000.0),
            "stop_loss_pct": c.stop_loss_pct,
            "isinma_gun": c.range_warmup_days if warmup_info["kullanildi"] else None,
        },
    )


def _compute_metrics(
    equity: pd.Series, df: pd.DataFrame, realized: float, unrealized: float,
    remaining_qty: float, remaining_value: float, remaining_cost: float,
    total_cost: float, n_pairs: int, n_trades: int, skipped_buys: int,
    grid_shifts: int, below_lower: int, above_upper: int, range_broken: bool,
    levels: np.ndarray, c: GridConfig, n: int,
) -> dict[str, Any]:
    """Dürüstlük kalemleri dahil özet metrikleri hesaplar."""
    capital = c.total_capital
    total_pnl = realized + unrealized
    dd = equity / equity.cummax() - 1.0

    bh = buy_and_hold(df, c)
    step = grid_step_pct(levels)
    roundtrip = 2.0 * (c.commission_rate + c.slippage_bps / 10_000.0)

    return {
        # --- Dürüstlük üçlüsü: bunlar AYRI raporlanır ---------------------- #
        "gerceklesmis_kar": realized,
        "gerceklesmis_kar_oran": realized / capital,
        "gerceklesmemis_kz": unrealized,
        "gerceklesmemis_kz_oran": unrealized / capital,
        "toplam_kz": total_pnl,
        "toplam_getiri": total_pnl / capital,
        # --- Elde kalan envanter (grid'in gizli riski) --------------------- #
        "elde_kalan_miktar": remaining_qty,
        "elde_kalan_deger": remaining_value,
        "elde_kalan_maliyet": remaining_cost,
        "sermayenin_coinde_orani": remaining_value / capital,
        # --- Aralık sağlığı ------------------------------------------------ #
        "aralik_kirildi": bool(range_broken),
        "alt_sinir_alti_bar": int(below_lower),
        "ust_sinir_ustu_bar": int(above_upper),
        "aralik_disi_sure_orani": (below_lower + above_upper) / max(n, 1),
        "grid_kaydirma": int(grid_shifts),
        # --- İşlem ve maliyet ---------------------------------------------- #
        "tamamlanan_cift": int(n_pairs),
        "islem_sayisi": int(n_trades),
        "atlanan_alis_nakit_yok": int(skipped_buys),
        "toplam_maliyet": total_cost,
        "maliyetin_sermayeye_orani": total_cost / capital,
        "kademe_adimi_yuzde": step,
        "gidis_donus_maliyet": roundtrip,
        "adim_maliyeti_kaciyor": step / roundtrip if roundtrip > 0 else float("nan"),
        # --- Risk ----------------------------------------------------------- #
        "max_drawdown": float(dd.min()),
        "son_sermaye": float(equity.iloc[-1]),
        # --- Ölçüt ---------------------------------------------------------- #
        "al_ve_tut_getiri": bh["getiri"],
        "al_ve_tut_max_dd": bh["max_drawdown"],
        "gridin_al_tut_ustunlugu": total_pnl / capital - bh["getiri"],
    }


def buy_and_hold(df: pd.DataFrame, cfg: GridConfig | None = None) -> dict[str, float]:
    """Al-ve-tut ölçütü (tek giriş maliyetiyle).

    Args:
        df: OHLCV verisi.
        cfg: Grid konfigürasyonu (maliyet için).

    Returns:
        ``getiri`` ve ``max_drawdown``.
    """
    c = cfg or CONFIG.grid
    entry_cost = c.commission_rate + c.slippage_bps / 10_000.0
    curve = df["close"] / df["close"].iloc[0] * (1.0 - entry_cost)
    dd = curve / curve.cummax() - 1.0
    return {"getiri": float(curve.iloc[-1] - 1.0), "max_drawdown": float(dd.min())}


# --------------------------------------------------------------------------- #
# Rejim analizi (dönem seçme yanlılığına karşı)
# --------------------------------------------------------------------------- #


def label_trend(first: float, last: float, threshold: float) -> str:
    """Bir pencerenin net yönünü etiketler.

    Args:
        first: Pencerenin ilk kapanışı.
        last: Pencerenin son kapanışı.
        threshold: "Yatay" sayılacak azami mutlak değişim.

    Returns:
        ``"yukselis"``, ``"dusus"`` veya ``"yatay"``.
    """
    change = last / first - 1.0 if first else 0.0
    if change > threshold:
        return "yukselis"
    if change < -threshold:
        return "dusus"
    return "yatay"


def regime_analysis(
    df: pd.DataFrame,
    equity: pd.Series,
    cfg: GridConfig | None = None,
) -> pd.DataFrame:
    """Stratejiyi alt dönemlere bölerek rejim bazında raporlar.

    Tek bir 60 günlük toplam sayı, dönem seçme yanlılığını (period selection
    bias) gizler: grid yatay piyasada para kazanır, düşüşte kaybeder. Toplam
    rakam hangisinin baskın olduğuna bağlıdır ve gelecek için hiçbir şey
    söylemez. Bu yüzden her pencere ayrı raporlanır ve fiyatın o pencaredeki
    net yönü etiketlenir.

    Args:
        df: OHLCV verisi.
        equity: Strateji sermaye eğrisi.
        cfg: Grid konfigürasyonu.

    Returns:
        Pencere başına getiri, drawdown, fiyat değişimi ve rejim etiketi.
    """
    c = cfg or CONFIG.grid
    if df.empty or equity.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouper = df["close"].resample(c.regime_window, label="left", closed="left")
    for start, chunk in grouper:
        if len(chunk) < 2:
            continue
        eq = equity.loc[chunk.index[0] : chunk.index[-1]]
        if len(eq) < 2:
            continue
        eq_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        dd = float((eq / eq.cummax() - 1.0).min())
        px_change = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
        rows.append({
            "pencere_baslangic": start,
            "bar": int(len(chunk)),
            "fiyat_degisim": px_change,
            "rejim": label_trend(float(chunk.iloc[0]), float(chunk.iloc[-1]), c.sideways_threshold),
            "strateji_getiri": eq_ret,
            "strateji_max_dd": dd,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("pencere_baslangic").reset_index(drop=True)


def regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    """Rejim bazında ortalama davranışı özetler.

    Args:
        regimes: :func:`regime_analysis` çıktısı.

    Returns:
        Rejim etiketi başına pencere sayısı, ortalama getiri ve drawdown.
    """
    if regimes.empty:
        return pd.DataFrame()
    g = regimes.groupby("rejim")
    return pd.DataFrame({
        "pencere": g.size(),
        "ort_getiri": g["strateji_getiri"].mean(),
        "medyan_getiri": g["strateji_getiri"].median(),
        "en_kotu": g["strateji_getiri"].min(),
        "ort_max_dd": g["strateji_max_dd"].mean(),
        "pozitif_oran": g["strateji_getiri"].apply(lambda s: float((s > 0).mean())),
    }).round(4)


# --------------------------------------------------------------------------- #
# Raporlama
# --------------------------------------------------------------------------- #


def print_report(result: GridResult, title: str = "GRID BACKTEST") -> None:
    """Sonucu, gizlenmiş risk kalmayacak biçimde ekrana basar.

    Args:
        result: Backtest sonucu.
        title: Başlık.
    """
    m = result.metrics
    print("\n" + "=" * 78)
    print(f"{title}  |  mod={result.config['mode']}  kademe={result.config['n_levels']}")
    print("=" * 78)
    print(f"  Aralik            : {result.config['alt_sinir']:,.0f} - {result.config['ust_sinir']:,.0f}")
    print(f"  Kademe adimi      : {m['kademe_adimi_yuzde'] * 100:.3f}%  "
          f"(gidis-donus maliyet {m['gidis_donus_maliyet'] * 100:.2f}%, "
          f"adim/maliyet = {m['adim_maliyeti_kaciyor']:.2f}x)")
    if m["adim_maliyeti_kaciyor"] < 1.5:
        print("    UYARI: adim maliyete cok yakin; her cift kari maliyetle eriyor.")

    print("\n  --- KAR/ZARAR AYRISTIRMASI (grid'in gizli riski burada) ---")
    print(f"  Gerceklesmis kar  : {m['gerceklesmis_kar']:>14,.2f} TL  ({m['gerceklesmis_kar_oran']:+.2%})")
    print(f"  Gerceklesmemis K/Z: {m['gerceklesmemis_kz']:>14,.2f} TL  ({m['gerceklesmemis_kz_oran']:+.2%})")
    print(f"  TOPLAM K/Z        : {m['toplam_kz']:>14,.2f} TL  ({m['toplam_getiri']:+.2%})")
    if m["gerceklesmis_kar"] > 0 and m["toplam_kz"] < 0:
        print("    DIKKAT: gerceklesmis kar POZITIF ama toplam NEGATIF.")
        print("    Yalnizca 'gerceklesmis kar' raporlansaydi strateji karli gorunecekti.")

    print("\n  --- ELDE KALAN ENVANTER ---")
    print(f"  Miktar            : {m['elde_kalan_miktar']:.8f}")
    print(f"  Degeri / maliyeti : {m['elde_kalan_deger']:,.2f} / {m['elde_kalan_maliyet']:,.2f} TL")
    print(f"  Sermayenin coinde : {m['sermayenin_coinde_orani']:.1%}")

    print("\n  --- ARALIK SAGLIGI ---")
    print(f"  ARALIK KIRILDI    : {'EVET — strateji fiilen coktu' if m['aralik_kirildi'] else 'hayir'}")
    print(f"  Alt sinir alti    : {m['alt_sinir_alti_bar']:,} bar")
    print(f"  Ust sinir ustu    : {m['ust_sinir_ustu_bar']:,} bar")
    print(f"  Aralik disi sure  : {m['aralik_disi_sure_orani']:.1%}")
    if result.config["mode"] == "trailing":
        print(f"  Grid kaydirma     : {m['grid_kaydirma']}")

    print("\n  --- ISLEM VE MALIYET ---")
    print(f"  Tamamlanan cift   : {m['tamamlanan_cift']:,}")
    print(f"  Toplam islem      : {m['islem_sayisi']:,}")
    print(f"  Nakit yetmedi     : {m['atlanan_alis_nakit_yok']:,} alis atlandi")
    print(f"  Toplam maliyet    : {m['toplam_maliyet']:,.2f} TL ({m['maliyetin_sermayeye_orani']:.2%})")

    print("\n  --- RISK VE OLCUT ---")
    print(f"  Max drawdown      : {m['max_drawdown']:.2%}")
    print(f"  Al-ve-tut getiri  : {m['al_ve_tut_getiri']:+.2%}  (maxDD {m['al_ve_tut_max_dd']:.2%})")
    print(f"  Grid ustunlugu    : {m['gridin_al_tut_ustunlugu']:+.2%} "
          f"({'grid daha iyi' if m['gridin_al_tut_ustunlugu'] > 0 else 'sadece tutmak daha iyiydi'})")

    if not result.regimes.empty:
        print("\n  --- REJIM ANALIZI (donem secme yanliligina karsi) ---")
        print(regime_summary(result.regimes).to_string())


def plot_grid(result: GridResult, df: pd.DataFrame, path: str | Path) -> Path:
    """Fiyat + kademeler, sermaye eğrisi ve envanter grafiğini yazar.

    Args:
        result: Backtest sonucu.
        df: OHLCV verisi.
        path: Hedef PNG yolu.

    Returns:
        Yazılan dosyanın yolu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), height_ratios=[2, 1, 1], sharex=True)

    axes[0].plot(df.index, df["close"].to_numpy(), color="#2d3748", lw=0.8, label="Fiyat")
    for lvl in result.levels:
        axes[0].axhline(lvl, color="#a0aec0", lw=0.4, alpha=0.6)
    axes[0].axhline(result.levels[0], color="#c53030", lw=1.2, label="Alt sinir")
    axes[0].axhline(result.levels[-1], color="#2f855a", lw=1.2, label="Ust sinir")
    axes[0].set_title("Fiyat ve grid kademeleri")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result.equity.index, result.equity.to_numpy(), color="#2b6cb0")
    axes[1].axhline(CONFIG.grid.total_capital, color="#718096", ls="--", lw=0.8)
    axes[1].set_ylabel("Sermaye (TL)")
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(result.inventory.index, result.inventory.to_numpy(), 0,
                         color="#dd6b20", alpha=0.6)
    axes[2].set_ylabel("Envanter (coin)")
    axes[2].set_xlabel("Zaman")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


#: Grid testi için beklenen varsayılan veri dosyası (Binance prototip verisi).
DEFAULT_DATA = "research/data/ohlcv/BTCTRY_1m_binance.parquet"


def main(argv: list[str] | None = None) -> int:
    """Komut satırı: veriyi yükle, 1 saatlik bara indir, iki modu da raporla.

    Args:
        argv: Argümanlar (test için).

    Returns:
        Çıkış kodu.
    """
    import argparse
    import copy

    from .data import generate_synthetic_ohlcv, load_ohlcv, resample_ohlcv

    ap = argparse.ArgumentParser(description="Grid stratejisi backtest (maliyetli, durust).")
    ap.add_argument("--data", type=str, default=DEFAULT_DATA, help="OHLCV parquet/csv yolu.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Veri yoksa sentetik seriyle GOSTERIM yap (gercek sonuc DEGIL).")
    ap.add_argument("--resample", type=str, default="1h",
                    help="Hedef bar (varsayilan 1h; 1m grid'de asiri islem/maliyet olur).")
    ap.add_argument("--levels", type=int, default=None, help="Kademe sayisi.")
    ap.add_argument("--capital", type=float, default=None, help="Toplam sermaye.")
    ap.add_argument("--lower", type=float, default=None, help="Alt sinir fiyati.")
    ap.add_argument("--upper", type=float, default=None, help="Ust sinir fiyati.")
    ap.add_argument("--plot", type=str, default=None, help="Grafik PNG yolu.")
    args = ap.parse_args(argv)

    path = Path(args.data)
    if path.exists():
        df = load_ohlcv(path)
        source = str(path)
    elif args.synthetic:
        print(
            "UYARI: gercek veri yok, SENTETIK seri kullaniliyor.\n"
            "  Bu bir GOSTERIMDIR; sonuclar hicbir piyasa hakkinda bilgi vermez.\n"
        )
        df = generate_synthetic_ohlcv(n_bars=60 * 1440, seed=CONFIG.seed)
        source = "sentetik"
    else:
        print(
            f"Veri bulunamadi: {path}\n\n"
            "Bu dosya Binance indiricisiyle uretilir:\n"
            "    python -m research.tools.fetch_binance --days 60\n"
            "Indirme bu ortamda egress politikasi nedeniyle yapilamadi;\n"
            "kendi makinenizde calistirip dosyayi olusturun.\n\n"
            "Yalnizca motoru gormek icin: --synthetic ekleyin."
        )
        return 1

    if args.resample:
        before = len(df)
        df = resample_ohlcv(df, args.resample)
        print(f"Veri: {source}\n  {before:,} bar -> {len(df):,} bar ({args.resample})")
        print(f"  {df.index[0]} -> {df.index[-1]}")

    cfg = copy.deepcopy(CONFIG.grid)
    if args.levels:
        cfg.n_levels = args.levels
    if args.capital:
        cfg.total_capital = args.capital
    if args.lower:
        cfg.lower_price = args.lower
    if args.upper:
        cfg.upper_price = args.upper

    table, results = compare_modes(df, cfg)
    for mode, res in results.items():
        print_report(res, title=f"GRID BACKTEST — {source}")

    print("\n" + "=" * 78)
    print("MOD KARSILASTIRMASI")
    print("=" * 78)
    print(table.to_string(index=False))

    print("\n--- REJIM DETAYI (static) ---")
    reg = results["static"].regimes
    if not reg.empty:
        print(reg.round(4).to_string(index=False))

    if args.plot:
        p = plot_grid(results["static"], df, args.plot)
        print(f"\nGrafik: {p}")
    return 0


def compare_modes(
    df: pd.DataFrame,
    cfg: GridConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, GridResult]]:
    """Sabit ve hareketli aralık modlarını AYRI çalıştırıp karşılaştırır.

    Args:
        df: OHLCV verisi.
        cfg: Temel konfigürasyon.

    Returns:
        ``(karsilastirma_tablosu, {mod: sonuc})``.
    """
    import copy

    base = cfg or CONFIG.grid
    results: dict[str, GridResult] = {}
    rows: list[dict[str, Any]] = []
    for mode in ("static", "trailing"):
        cc = copy.deepcopy(base)
        cc.mode = mode
        res = run_grid_backtest(df, cc)
        results[mode] = res
        m = res.metrics
        rows.append({
            "mod": mode,
            "gerceklesmis": round(m["gerceklesmis_kar_oran"], 4),
            "gerceklesmemis": round(m["gerceklesmemis_kz_oran"], 4),
            "toplam": round(m["toplam_getiri"], 4),
            "max_dd": round(m["max_drawdown"], 4),
            "cift": m["tamamlanan_cift"],
            "aralik_kirildi": m["aralik_kirildi"],
            "aralik_disi": round(m["aralik_disi_sure_orani"], 4),
            "al_tut_farki": round(m["gridin_al_tut_ustunlugu"], 4),
        })
    return pd.DataFrame(rows), results


if __name__ == "__main__":
    raise SystemExit(main())
