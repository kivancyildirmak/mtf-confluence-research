"""Etiketleme: triple-barrier, meta-labeling ve örnek benzersizliği ağırlıkları.

Yaklaşım López de Prado, *Advances in Financial Machine Learning* (Bölüm 3-4)
temellidir.

**Neden sabit ufuklu etiket kullanmıyoruz?** "60 bar sonraki getirinin işareti"
gibi etiketler, gerçek işlemin nasıl kapandığını (zarar-kes, kâr-al) yansıtmaz.
Triple-barrier ise pozisyonun gerçekte hangi bariyerle kapandığını etiketler.

**SIZINTI POLİTİKASI.** Etiketler tanımı gereği GELECEĞE bakar — sorun bu değil,
sorun etiketin geleceğe baktığı pencerenin eğitim/test bölünmesine sızmasıdır.
Bu yüzden her olay için etiketin kapandığı zaman (``t1``) döndürülür ve
:mod:`validation` bu bilgiyle purging + embargo uygular. ``t1``'i taşımadan
yapılan hiçbir çapraz doğrulama güvenilir değildir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, LabelConfig

#: Bariyer tipleri için sayısal kodlar.
BARRIER_PT: int = 1
BARRIER_SL: int = -1
BARRIER_VERTICAL: int = 0


# --------------------------------------------------------------------------- #
# Hedef volatilite ve olay örnekleme
# --------------------------------------------------------------------------- #


def get_target_volatility(
    close: pd.Series,
    span: int | None = None,
    horizon_bars: int | None = None,
    cfg: LabelConfig | None = None,
) -> pd.Series:
    """Bariyer ölçeklemesi için ileriye dönük ufka göre volatilite tahmini.

    Bar başına EWM standart sapma hesaplanır ve ``sqrt(horizon)`` ile dikey
    bariyer ufkuna ölçeklenir. Böylece bariyerler "bu ufukta tipik olarak ne
    kadar oynar?" sorusunun cevabına oturur.

    Args:
        close: Kapanış serisi.
        span: EWM span'ı (bar). ``None`` ise config'ten.
        horizon_bars: Dikey bariyer ufku. ``None`` ise config'ten.
        cfg: Etiketleme konfigürasyonu.

    Returns:
        Ufka ölçeklenmiş volatilite serisi (oran cinsinden).

    Notes:
        SIZINTI: ``ewm`` yalnızca geçmişe bakar. Volatiliteyi tüm örnek üzerinden
        (``close.std()``) hesaplamak klasik ve sinsi bir sızıntı hatasıdır.
    """
    c = cfg or CONFIG.labeling
    sp = span if span is not None else c.vol_span
    h = horizon_bars if horizon_bars is not None else c.vertical_bars

    ret = np.log(close).diff()
    bar_vol = ret.ewm(span=sp, adjust=False, min_periods=sp // 2).std()
    return (bar_vol * np.sqrt(float(h))).rename("target_vol")


def cusum_filter(
    close: pd.Series,
    threshold: pd.Series | float,
) -> pd.DatetimeIndex:
    """Simetrik CUSUM olay filtresi (LdP Böl. 2.5.2.1).

    Her barı bir olay saymak yerine, kümülatif log-getiri belirli bir eşiği
    aştığında olay üretir. Faydası:

    * Örnek çakışmasını (overlapping labels) ciddi biçimde azaltır.
    * Modeli "hiçbir şey olmayan" barlarla eğitmekten kurtarır.

    Args:
        close: Kapanış serisi.
        threshold: Skaler eşik veya bara göre değişen eşik serisi (genelde
            hedef volatilitenin katı).

    Returns:
        Olay zamanlarının :class:`pandas.DatetimeIndex`'i.

    Notes:
        SIZINTI: Filtre yalnızca geçmiş getirileri biriktirir; eşik seri olarak
        verildiğinde o da kayan pencereden gelmelidir.
    """
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    if isinstance(threshold, (int, float)):
        thr = np.full(len(close), float(threshold))
    else:
        thr = pd.Series(threshold).reindex(close.index).to_numpy(dtype="float64")

    events: list[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for i in range(len(ret)):
        h = thr[i]
        if not np.isfinite(h) or h <= 0.0:
            continue
        s_pos = max(0.0, s_pos + ret[i])
        s_neg = min(0.0, s_neg + ret[i])
        if s_pos > h:
            s_pos = 0.0
            events.append(i)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(i)
    return close.index[events]


def get_vertical_barriers(
    index: pd.DatetimeIndex,
    event_index: pd.DatetimeIndex,
    vertical_bars: int,
) -> pd.Series:
    """Her olay için dikey (süre) bariyerin zamanını döndürür.

    Args:
        index: Tüm bar zaman damgaları.
        event_index: Olay zaman damgaları.
        vertical_bars: Kaç bar sonra pozisyonun zorla kapanacağı.

    Returns:
        Olay zamanından dikey bariyer zamanına eşleyen seri. Ufku veri sonuna
        SIĞMAYAN olaylar için ``NaT`` döner.

    Notes:
        Sığmayan olayların bariyerini son bara "kırpmak" sinsi bir hatadır: bu
        olaylar sistematik olarak daha kısa ufuklu ve daha küçük getirili
        görünür, yani örneklemin sonu yapay biçimde kolaylaşır. Doğrusu onları
        ``NaT`` işaretleyip etiketleme dışında BIRAKMAKTIR.
    """
    pos = index.get_indexer(event_index)
    end_pos = pos + int(vertical_bars)
    valid = end_pos <= len(index) - 1
    out = pd.Series(pd.NaT, index=event_index, name="t1", dtype=index.dtype)
    if valid.any():
        out.loc[event_index[valid]] = index[end_pos[valid]]
    return out


# --------------------------------------------------------------------------- #
# Triple-barrier çekirdeği
# --------------------------------------------------------------------------- #


def _triple_barrier_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ev_pos: np.ndarray,
    end_pos: np.ndarray,
    pt: np.ndarray,
    sl: np.ndarray,
    side: np.ndarray,
    use_intrabar: bool,
    sl_wins_tie: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bariyer taramasının saf-NumPy çekirdeği (pozisyon indeksleriyle çalışır).

    Args:
        close: Kapanış dizisi.
        high: En yüksek dizisi.
        low: En düşük dizisi.
        ev_pos: Olay barlarının konum indeksleri.
        end_pos: Dikey bariyer konum indeksleri.
        pt: Kâr-al bariyeri (oran, pozitif).
        sl: Zarar-kes bariyeri (oran, pozitif).
        side: İşlem yönü (+1 alış, -1 satış).
        use_intrabar: Bariyer teması high/low ile mi ölçülsün.
        sl_wins_tie: Aynı barda ikisi de değerse zarar-kes öncelikli mi.

    Returns:
        ``(touch_pos, ret, barrier_type)`` üçlüsü.
    """
    n_ev = len(ev_pos)
    touch = np.empty(n_ev, dtype=np.int64)
    rets = np.empty(n_ev, dtype=np.float64)
    btype = np.empty(n_ev, dtype=np.int8)

    for k in range(n_ev):
        p = ev_pos[k]
        end = end_pos[k]
        s = side[k]
        entry = close[p]
        # Yön-bağımsız bariyer fiyatları.
        tp_price = entry * (1.0 + s * pt[k])
        sl_price = entry * (1.0 - s * sl[k])

        hit_pos = end
        hit_price = close[end]
        hit_type = BARRIER_VERTICAL

        # Tarama p+1'den başlar: olay barının KENDİ hareketiyle işlem
        # kapatılamaz (o bar zaten kapanmıştır, giriş onun kapanışındadır).
        for j in range(p + 1, end + 1):
            if use_intrabar:
                hi, lo = high[j], low[j]
            else:
                hi = lo = close[j]

            if s > 0:
                tp_hit = hi >= tp_price
                sl_hit = lo <= sl_price
            else:
                tp_hit = lo <= tp_price
                sl_hit = hi >= sl_price

            if tp_hit and sl_hit:
                # Bar içi sıralama bilinmiyor: kötümser varsayım (SL önce).
                if sl_wins_tie:
                    hit_pos, hit_price, hit_type = j, sl_price, BARRIER_SL
                else:
                    hit_pos, hit_price, hit_type = j, tp_price, BARRIER_PT
                break
            if tp_hit:
                hit_pos, hit_price, hit_type = j, tp_price, BARRIER_PT
                break
            if sl_hit:
                hit_pos, hit_price, hit_type = j, sl_price, BARRIER_SL
                break

        touch[k] = hit_pos
        rets[k] = s * (hit_price / entry - 1.0)
        btype[k] = hit_type

    return touch, rets, btype


def get_triple_barrier_labels(
    df: pd.DataFrame,
    event_index: pd.DatetimeIndex | None = None,
    target_vol: pd.Series | None = None,
    side: pd.Series | None = None,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    """Triple-barrier etiketleri üretir.

    Üç bariyer:

    1. **Üst (kâr-al)** — ``pt_mult * target_vol`` kadar yukarıda,
    2. **Alt (zarar-kes)** — ``sl_mult * target_vol`` kadar aşağıda,
    3. **Dikey (süre)** — ``vertical_bars`` bar sonra.

    Etiket, İLK DEĞİLEN bariyere göre belirlenir.

    Args:
        df: OHLCV verisi.
        event_index: Etiketlenecek olay zamanları. ``None`` ise CUSUM filtresi
            ile üretilir.
        target_vol: Bariyer ölçeği. ``None`` ise :func:`get_target_volatility`.
        side: İşlem yönü serisi (meta-labeling için birincil modelin yönü).
            ``None`` ise tüm olaylar alış (+1) varsayılır ve etiket YÖN
            (-1/0/+1) olarak döner.
        cfg: Etiketleme konfigürasyonu.

    Returns:
        Olay indeksli DataFrame:

        ``t1`` (bariyerin değildiği zaman), ``ret`` (yön düzeltilmiş ham getiri,
        maliyetsiz), ``barrier`` (1=kâr-al, -1=zarar-kes, 0=süre),
        ``label`` (yönlü etiket veya meta etiket), ``target`` (hedef vol),
        ``side`` (yön), ``holding_bars`` (pozisyonda kalınan bar sayısı).

    Notes:
        ``ret`` içinde komisyon/slippage YOKTUR. Maliyetler kasıtlı olarak
        :mod:`backtest` katmanına bırakılmıştır; etiket ile icra varsayımlarını
        birbirine karıştırmamak için.

        SIZINTI: Dönen ``t1`` sütunu ZORUNLUDUR — :mod:`validation` purging'i
        bu sütuna dayanır. ``t1``'i atıp sadece ``label`` ile CV yapmak, ufku
        60 bar olan etiketlerde en az 60 barlık sızıntı demektir.
    """
    c = cfg or CONFIG.labeling
    close = df["close"]

    tv = target_vol if target_vol is not None else get_target_volatility(close, cfg=c)

    if event_index is None:
        event_index = cusum_filter(close, tv * c.cusum_threshold_mult)

    events = pd.DatetimeIndex(event_index)
    # Hedef vol'ü olmayan (ısınma dönemi) olayları at.
    tv_ev = tv.reindex(events)
    keep = tv_ev.notna() & (tv_ev > c.min_ret)
    events = events[keep.to_numpy()]
    if len(events) == 0:
        return _empty_label_frame()

    tv_ev = tv.reindex(events)
    t1 = get_vertical_barriers(df.index, events, c.vertical_bars)

    # Ufku veri sonuna sığmayan olaylar elenir (kırpılmış etiket = yanlı etiket).
    fits = t1.notna()
    if not fits.all():
        events = events[fits.to_numpy()]
        if len(events) == 0:
            return _empty_label_frame()
        t1 = t1.loc[events]
        tv_ev = tv_ev.loc[events]

    if side is None:
        side_ev = pd.Series(1.0, index=events, name="side")
        is_meta = False
    else:
        side_ev = pd.Series(side).reindex(events).astype("float64")
        # Yönü tanımsız veya 0 olan olaylar işleme girmez.
        valid = side_ev.notna() & (side_ev != 0.0)
        events = events[valid.to_numpy()]
        if len(events) == 0:
            return _empty_label_frame()
        side_ev = side_ev.loc[events]
        tv_ev = tv_ev.loc[events]
        t1 = t1.loc[events]
        is_meta = True

    pt_mult, sl_mult = c.pt_sl
    pt = (tv_ev * pt_mult).to_numpy(dtype="float64")
    sl = (tv_ev * sl_mult).to_numpy(dtype="float64")

    ev_pos = df.index.get_indexer(events)
    end_pos = df.index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))

    touch_pos, rets, btype = _triple_barrier_core(
        close.to_numpy(dtype="float64"),
        df["high"].to_numpy(dtype="float64"),
        df["low"].to_numpy(dtype="float64"),
        ev_pos.astype(np.int64),
        end_pos.astype(np.int64),
        pt,
        sl,
        side_ev.to_numpy(dtype="float64"),
        c.use_intrabar_extremes,
        c.tie_break == "sl",
    )

    out = pd.DataFrame(
        {
            "t1": df.index[touch_pos],
            "ret": rets,
            "barrier": btype.astype("int64"),
            "target": tv_ev.to_numpy(),
            "side": side_ev.to_numpy(),
            "holding_bars": (touch_pos - ev_pos).astype("int64"),
        },
        index=events,
    )
    out.index.name = "event_time"

    if is_meta:
        # Meta-label: "birincil modelin yönüne GÜVENİP işleme girmeli miyim?"
        out["label"] = (out["ret"] > 0.0).astype("int64")
    else:
        # Yön etiketi: süre bariyeriyle kapananlar 0 (kararsız) sayılır.
        out["label"] = np.where(
            out["barrier"] == BARRIER_VERTICAL, 0, out["barrier"]
        ).astype("int64")
    return out


def _empty_label_frame() -> pd.DataFrame:
    """Hiç olay kalmadığında dönen boş ama şema-uyumlu DataFrame."""
    idx = pd.DatetimeIndex([], name="event_time", tz="UTC")
    return pd.DataFrame(
        {
            "t1": pd.Series(dtype="datetime64[ns, UTC]"),
            "ret": pd.Series(dtype="float64"),
            "barrier": pd.Series(dtype="int64"),
            "target": pd.Series(dtype="float64"),
            "side": pd.Series(dtype="float64"),
            "holding_bars": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="int64"),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Birincil model (yön) ve meta-labeling
# --------------------------------------------------------------------------- #


def primary_side_rule(
    df: pd.DataFrame,
    fast: int = 15,
    slow: int = 60,
    neutral_z: float = 0.0,
) -> pd.Series:
    """Basit, kural tabanlı BİRİNCİL model: yön üretir (+1 / -1 / 0).

    Neden ML değil de kural? Meta-labeling'de birincil modelin yönü, meta
    modelin eğitildiği veriden BAĞIMSIZ olmalıdır. Birincil model de aynı veri
    üzerinde ML ile eğitilirse, örneklem-içi (in-sample) yön tahminleri meta
    etiketlere sızar ve meta model gerçekte olmayan bir kenar (edge) görür.
    Deterministik bir kural bu sızıntıyı tanımı gereği ortadan kaldırır.

    Args:
        df: OHLCV verisi.
        fast: Hızlı EMA periyodu.
        slow: Yavaş EMA periyodu.
        neutral_z: Bu eşiğin altındaki sinyal gücü 0 (işlem yok) sayılır.

    Returns:
        Her bar için ``+1`` (alış), ``-1`` (satış), ``0`` (yön yok) serisi.

    Notes:
        SIZINTI: EMA'lar ``adjust=False`` ile özyinelemelidir ve yalnızca
        geçmişe bakar. Sinyal bar ``t``'nin kapanışında bilinir.
    """
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    spread = (ema_fast - ema_slow) / close
    # Sinyal gücünü kendi geçmişine göre normalize et (kayan z-skoru).
    z = spread / spread.rolling(slow * 4, min_periods=slow).std(ddof=0)
    side = np.sign(z.where(z.abs() > neutral_z, 0.0))
    return side.fillna(0.0).rename("side")


def get_meta_labels(
    df: pd.DataFrame,
    primary_side: pd.Series,
    event_index: pd.DatetimeIndex | None = None,
    target_vol: pd.Series | None = None,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    """Meta-labeling: birincil modelin YÖNÜNÜ alıp "gir / girme" etiketi üretir.

    İkincil (meta) model yön tahmin etmez; yalnızca birincil modelin önerdiği
    işlemin kârlı olup olmayacağına karar verir. Faydası:

    * Hatalı pozitifleri (false positive) azaltır, isabet oranını yükseltir.
    * Çıktı bir OLASILIKTIR; :mod:`sizing` bunu doğrudan pozisyon boyutuna
      çevirebilir.

    Args:
        df: OHLCV verisi.
        primary_side: Birincil modelin yön serisi (bar bazında).
        event_index: Olay zamanları. ``None`` ise CUSUM ile üretilir.
        target_vol: Hedef volatilite serisi.
        cfg: Etiketleme konfigürasyonu.

    Returns:
        ``label ∈ {0, 1}`` içeren olay tablosu (bkz.
        :func:`get_triple_barrier_labels`).
    """
    c = cfg or CONFIG.labeling
    tv = target_vol if target_vol is not None else get_target_volatility(df["close"], cfg=c)
    if event_index is None:
        event_index = cusum_filter(df["close"], tv * c.cusum_threshold_mult)
    return get_triple_barrier_labels(
        df, event_index=event_index, target_vol=tv, side=primary_side, cfg=c
    )


# --------------------------------------------------------------------------- #
# Örnek benzersizliği ve ağırlıklar (LdP Böl. 4)
# --------------------------------------------------------------------------- #


def get_num_co_events(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.Series:
    """Her barda kaç etiketin AYNI ANDA açık olduğunu sayar.

    Çakışan etiketler bağımsız gözlem değildir; bu sayım, ağırlıklandırmanın
    temelidir.

    Args:
        bar_index: Tüm bar zaman damgaları.
        t1: Olay -> bariyer zamanı eşlemesi.

    Returns:
        Bar bazında eşzamanlı olay sayısı.
    """
    if len(t1) == 0:
        return pd.Series(0.0, index=bar_index, name="co_events")
    start = bar_index.get_indexer(pd.DatetimeIndex(t1.index))
    end = bar_index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    # Fark dizisi + kümülatif toplam: O(n) sayım.
    delta = np.zeros(len(bar_index) + 1, dtype="float64")
    np.add.at(delta, start, 1.0)
    np.add.at(delta, end + 1, -1.0)
    counts = np.cumsum(delta)[: len(bar_index)]
    return pd.Series(counts, index=bar_index, name="co_events")


def get_average_uniqueness(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series,
    co_events: pd.Series | None = None,
) -> pd.Series:
    """Her etiketin ORTALAMA BENZERSİZLİĞİNİ hesaplar (0-1).

    Bir etiketin ömrü boyunca, o barlarda ortalama ``1 / eşzamanlı_olay_sayısı``
    değeridir. 1'e yakınsa etiket benzersizdir; 0'a yakınsa aynı bilgiyi
    paylaşan çok sayıda örnek vardır.

    Args:
        bar_index: Tüm bar zaman damgaları.
        t1: Olay -> bariyer zamanı eşlemesi.
        co_events: Önceden hesaplanmış eşzamanlılık sayımı.

    Returns:
        Olay indeksli ortalama benzersizlik serisi.
    """
    if len(t1) == 0:
        return pd.Series(dtype="float64", name="uniqueness")
    ce = co_events if co_events is not None else get_num_co_events(bar_index, t1)
    inv = np.where(ce.to_numpy() > 0, 1.0 / np.maximum(ce.to_numpy(), 1e-12), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    start = bar_index.get_indexer(pd.DatetimeIndex(t1.index))
    end = bar_index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    span = (end - start + 1).astype("float64")
    avg = (cum[end + 1] - cum[start]) / np.maximum(span, 1.0)
    return pd.Series(avg, index=t1.index, name="uniqueness")


def get_return_attribution_weights(
    close: pd.Series,
    t1: pd.Series,
    co_events: pd.Series | None = None,
) -> pd.Series:
    """Ağırlıkları etiketin ömrü boyunca ÜRETTİĞİ getiriye göre dağıtır.

    Fikir: büyük fiyat hareketi içeren örnekler daha bilgilendiricidir; ayrıca
    her barın getirisi o barda açık olan etiketler arasında paylaştırılır.

    Args:
        close: Kapanış serisi.
        t1: Olay -> bariyer zamanı eşlemesi.
        co_events: Eşzamanlılık sayımı.

    Returns:
        Ortalaması 1 olacak şekilde normalize edilmiş ağırlıklar.
    """
    if len(t1) == 0:
        return pd.Series(dtype="float64", name="weight")
    bar_index = close.index
    ce = co_events if co_events is not None else get_num_co_events(bar_index, t1)
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    denom = np.maximum(ce.to_numpy(), 1.0)
    contrib = ret / denom
    cum = np.concatenate([[0.0], np.cumsum(contrib)])
    start = bar_index.get_indexer(pd.DatetimeIndex(t1.index))
    end = bar_index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    w = np.abs(cum[end + 1] - cum[start])
    total = w.sum()
    w = w / total * len(w) if total > 0 else np.ones_like(w)
    return pd.Series(w, index=t1.index, name="weight")


def get_time_decay_weights(
    uniqueness: pd.Series,
    last_weight: float = 1.0,
) -> pd.Series:
    """Eski örneklerin ağırlığını doğrusal olarak azaltır (LdP time decay).

    Piyasa rejimleri değişir; çok eski örneklerin bugünkü rejimle ilgisi azdır.
    Sönüm, KÜMÜLATİF BENZERSİZLİK üzerinden uygulanır (takvim zamanı yerine),
    böylece yoğun örneklenmiş dönemler orantısız avantaj kazanmaz.

    Args:
        uniqueness: :func:`get_average_uniqueness` çıktısı.
        last_weight: En ESKİ örneğin alacağı ağırlık.
            ``1.0`` = sönüm yok, ``0.0`` = en eski örnek sıfırlanır,
            negatif değerler en eski örneklerin tamamen atılmasına yol açar.

    Returns:
        Sönüm katsayıları (olay indeksli).
    """
    if len(uniqueness) == 0:
        return pd.Series(dtype="float64", name="decay")
    cum = uniqueness.sort_index().cumsum()
    total = cum.iloc[-1]
    if total <= 0:
        return pd.Series(1.0, index=uniqueness.index, name="decay")
    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        slope = 1.0 / ((last_weight + 1.0) * total)
    const = 1.0 - slope * total
    decay = const + slope * cum
    decay[decay < 0] = 0.0
    return decay.reindex(uniqueness.index).rename("decay")


def get_sample_weights(
    close: pd.Series,
    events: pd.DataFrame,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    """Eğitimde kullanılacak nihai örnek ağırlıklarını üretir.

    Bileşenler:

    1. **Ortalama benzersizlik** — çakışan etiketlerin şişirdiği etkiyi kırar.
    2. **Getiri atfı** (opsiyonel) — bilgilendirici örneklere daha çok ağırlık.
    3. **Zaman sönümü** (opsiyonel) — eski rejimleri geri plana atar.

    Args:
        close: Kapanış serisi.
        events: :func:`get_triple_barrier_labels` çıktısı (``t1`` içermeli).
        cfg: Etiketleme konfigürasyonu.

    Returns:
        ``uniqueness``, ``ret_weight``, ``decay``, ``weight`` sütunlu tablo.
        ``weight`` ortalaması 1 olacak şekilde normalize edilir.

    Notes:
        Ağırlıklar SADECE eğitimde kullanılır. Performans metrikleri
        ağırlıksız hesaplanır; aksi halde raporlanan skor gerçek işlem
        sonuçlarını temsil etmez.
    """
    c = cfg or CONFIG.labeling
    if events.empty:
        return pd.DataFrame(
            columns=["uniqueness", "ret_weight", "decay", "weight"],
            index=events.index,
            dtype="float64",
        )

    t1 = events["t1"]
    co = get_num_co_events(close.index, t1)
    uniq = get_average_uniqueness(close.index, t1, co)

    if c.weight_by_return:
        rw = get_return_attribution_weights(close, t1, co)
    else:
        rw = pd.Series(1.0, index=events.index, name="weight")

    decay = get_time_decay_weights(uniq, c.time_decay_last_weight)

    weight = uniq * rw * decay
    mean = weight.mean()
    weight = weight / mean if mean > 0 else pd.Series(1.0, index=events.index)

    return pd.DataFrame(
        {
            "uniqueness": uniq,
            "ret_weight": rw,
            "decay": decay,
            "weight": weight.astype("float64"),
        }
    )


def label_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Etiket dağılımını ve bariyer kırılımını özetler (akıl sağlığı kontrolü).

    Args:
        events: Etiket tablosu.

    Returns:
        Tek satırlık özet tablo.

    Notes:
        Etiket dengesi aşırı bozuksa (örn. %95 tek sınıf) modelin "her zaman
        çoğunluk sınıfını söyleyerek" yüksek accuracy alması kaçınılmazdır.
        Bu yüzden accuracy'e asla tek başına güvenilmez.
    """
    if events.empty:
        return pd.DataFrame([{"n_events": 0}])
    n = len(events)
    row = {
        "n_events": n,
        "pozitif_oran": float((events["label"] == 1).mean()),
        "kar_al_oran": float((events["barrier"] == BARRIER_PT).mean()),
        "zarar_kes_oran": float((events["barrier"] == BARRIER_SL).mean()),
        "sure_bariyeri_oran": float((events["barrier"] == BARRIER_VERTICAL).mean()),
        "ort_tutma_bar": float(events["holding_bars"].mean()),
        "ort_getiri": float(events["ret"].mean()),
        "ort_hedef_vol": float(events["target"].mean()),
    }
    return pd.DataFrame([row])
