"""Hattın DAVRANIŞSAL testleri — özellikle sızıntı (look-ahead) güvenceleri.

Yorumda "sızıntı yok" yazmak yetmez; test etmek gerekir. Buradaki testler
hattın en kolay bozulan varsayımlarını doğrular:

* Özellikler nedenseldir (geleceği görmez).
* Etiketler olay barından SONRA başlayan bir pencereyi tarar.
* Purged K-Fold, test etiketleriyle çakışan eğitim örneklerini gerçekten atar.
* Maliyetler backtest sonucunu gerçekten kötüleştirir.
* Model kaydet/yükle döngüsü özellik şemasını korur.

Çalıştırma (pytest gerektirmez):

    python -m research.tests.test_pipeline

pytest kuruluysa:

    pytest research/tests/test_pipeline.py -v
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import run_backtest, total_cost_rate
from ..config import ResearchConfig
from ..data import (
    OHLCV_COLUMNS,
    fetch_orderbook,
    fetch_paribu_ohlcv,
    submit_order,
    generate_synthetic_ohlcv,
    generate_synthetic_orderbook,
    generate_synthetic_reference,
    load_ohlcv,
    save_ohlcv,
    validate_ohlcv,
)
from ..features import make_features, warmup_bars
from ..labeling import (
    cusum_filter,
    get_average_uniqueness,
    get_meta_labels,
    get_num_co_events,
    get_sample_weights,
    get_target_volatility,
    primary_side_rule,
)
from ..model import check_feature_parity, load_model, save_model, train_model
from ..sizing import compute_position_size, kelly_size, prob_to_bet_size
from ..validation import PurgedKFold, deflated_sharpe_ratio, expected_max_sharpe

_CFG = ResearchConfig()
_CFG.data.synthetic_bars = 12_000


def _sample_data() -> pd.DataFrame:
    """Testler için küçük ve deterministik bir OHLCV örneği."""
    return generate_synthetic_ohlcv(n_bars=12_000, seed=7, cfg=_CFG.data)


# --------------------------------------------------------------------------- #
# Veri katmanı
# --------------------------------------------------------------------------- #


def test_synthetic_data_is_valid() -> None:
    """Sentetik veri OHLCV tutarlılık kurallarını sağlamalı."""
    df = _sample_data()
    assert len(df) == 12_000
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] >= 0).all()


def test_synthetic_data_is_reproducible() -> None:
    """Aynı tohum aynı veriyi üretmeli (tekrarlanabilirlik)."""
    a = generate_synthetic_ohlcv(n_bars=500, seed=123, cfg=_CFG.data)
    b = generate_synthetic_ohlcv(n_bars=500, seed=123, cfg=_CFG.data)
    pd.testing.assert_frame_equal(a, b)


def test_csv_and_parquet_roundtrip() -> None:
    """Diske yazıp okumak veriyi bozmamalı."""
    df = generate_synthetic_ohlcv(n_bars=300, seed=5, cfg=_CFG.data)
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("t.csv", "t.parquet"):
            p = save_ohlcv(df, Path(tmp) / name)
            back = load_ohlcv(p)
            pd.testing.assert_frame_equal(df, back, check_freq=False, atol=1e-9)


def test_api_stubs_raise() -> None:
    """Bağlanmamış uçlar NotImplementedError atmalı.

    ``fetch_paribu_ohlcv`` artık stub DEĞİL (yerel poll-forward deposunu okur),
    bu yüzden buradan çıkarıldı; onun davranışı
    :func:`test_fetch_ohlcv_reports_missing_collection` ile test edilir.
    Emir defteri ucu Paribu'da yok, ``submit_order`` ise bilinçli olarak
    bağlanmadı.
    """
    for fn, args in ((fetch_orderbook, ("BTC_TL",)), (submit_order, ("BTC_TL", "buy", 1.0))):
        try:
            fn(*args)
        except NotImplementedError:
            continue
        raise AssertionError(f"{fn.__name__} NotImplementedError atmalıydı.")


def test_fetch_ohlcv_reports_missing_collection() -> None:
    """Hiç veri toplanmamışsa açıklayıcı bir hata verilmeli (sessiz boş DataFrame değil)."""
    import research.tools.collect_paribu as cp

    with tempfile.TemporaryDirectory() as tmp:
        orig = cp.CONFIG.collector.tick_dir
        cp.CONFIG.collector.tick_dir = str(Path(tmp) / "bos")
        try:
            fetch_paribu_ohlcv("BTC_TL")
        except FileNotFoundError as e:
            assert "collect" in str(e), "Hata mesajı ne yapılacağını söylemeli."
            return
        finally:
            cp.CONFIG.collector.tick_dir = orig
    raise AssertionError("Veri yokken FileNotFoundError beklenirdi.")


def test_validate_ohlcv_rejects_bad_schema() -> None:
    """Eksik sütun veya bozuk indeks reddedilmeli."""
    bad = pd.DataFrame({"open": [1.0], "close": [1.0]})
    try:
        validate_ohlcv(bad)
    except ValueError:
        return
    raise AssertionError("Eksik sütunlar ValueError atmalıydı.")


# --------------------------------------------------------------------------- #
# SIZINTI: özellik nedenselliği
# --------------------------------------------------------------------------- #


def test_features_are_causal() -> None:
    """EN KRİTİK TEST: özellikler geleceği görmemeli.

    Yöntem: Seriyi ``k``. bardan kesip özellik üretiyoruz. Kesilmiş seride
    hesaplanan SON satır, tüm seride hesaplanan AYNI satırla birebir aynı
    olmalıdır. Aynı değilse, o özellik gelecekteki barlara bakıyor demektir
    (``shift(-1)``, ``center=True``, global ortalama/std gibi klasik hatalar).
    """
    df = _sample_data()
    ob = generate_synthetic_orderbook(df, seed=2)
    ref = generate_synthetic_reference(df, seed=3)

    full = make_features(df, orderbook=ob, reference=ref, cfg=_CFG.features)

    for k in (8_000, 9_500, 11_000):
        cut = df.iloc[: k + 1]
        partial = make_features(
            cut,
            orderbook=ob.iloc[: k + 1],
            reference=ref.iloc[: k + 1],
            cfg=_CFG.features,
        )
        a = full.iloc[k]
        b = partial.iloc[-1]
        assert list(a.index) == list(b.index), "Özellik şeması kesitte değişmiş."
        diff = (a - b).abs()
        both_nan = a.isna() & b.isna()
        bad = diff[~both_nan & (diff > 1e-9)]
        assert bad.empty, f"Geleceğe bakan özellikler (bar {k}): {list(bad.index)}"


def test_feature_schema_is_stable_without_optional_sources() -> None:
    """Emir defteri/referans yokken de şema AYNI kalmalı (yalnızca NaN olur)."""
    df = _sample_data().iloc[:3_000]
    with_extras = make_features(
        df,
        orderbook=generate_synthetic_orderbook(df, seed=2),
        reference=generate_synthetic_reference(df, seed=3),
        cfg=_CFG.features,
    )
    without = make_features(df, cfg=_CFG.features)
    assert list(with_extras.columns) == list(without.columns)
    assert without["mk_spread_rel"].isna().all()
    assert without["xa_ref_ret_1"].isna().all()


def test_features_have_no_infinities() -> None:
    """Sonsuz değerler NaN'a çevrilmiş olmalı."""
    df = _sample_data().iloc[:3_000]
    X = make_features(df, cfg=_CFG.features)
    assert not np.isinf(X.to_numpy()).any()


# --------------------------------------------------------------------------- #
# SIZINTI: etiketleme
# --------------------------------------------------------------------------- #


def test_labels_look_forward_only_after_event() -> None:
    """Etiketin bariyer zamanı olay zamanından SONRA olmalı."""
    df = _sample_data()
    events = _events(df)
    assert (pd.DatetimeIndex(events["t1"]) > events.index).all()
    assert (events["holding_bars"] > 0).all()
    assert events["holding_bars"].max() <= _CFG.labeling.vertical_bars


def test_meta_labels_are_binary_and_match_returns() -> None:
    """Meta etiket, yön düzeltilmiş getirinin işaretiyle tutarlı olmalı."""
    df = _sample_data()
    events = _events(df)
    assert set(events["label"].unique()) <= {0, 1}
    assert ((events["ret"] > 0) == (events["label"] == 1)).all()
    assert set(np.unique(events["side"])) <= {-1.0, 1.0}


def test_sample_weights_reflect_overlap() -> None:
    """Çakışan etiketlerde benzersizlik 1'in altında olmalı."""
    df = _sample_data()
    events = _events(df)
    w = get_sample_weights(df["close"], events, cfg=_CFG.labeling)
    assert (w["uniqueness"] > 0).all()
    assert (w["uniqueness"] <= 1.0 + 1e-9).all()
    assert w["uniqueness"].mean() < 1.0, "Çakışma varken benzersizlik 1 olamaz."
    assert abs(w["weight"].mean() - 1.0) < 1e-6, "Ağırlıklar 1 ortalamaya normalize edilmeli."


def test_co_events_counting() -> None:
    """Eşzamanlılık sayımı elle doğrulanabilir bir örnekte doğru olmalı."""
    idx = pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC")
    t1 = pd.Series(
        [idx[3], idx[5], idx[9]], index=pd.DatetimeIndex([idx[0], idx[2], idx[6]])
    )
    co = get_num_co_events(idx, t1)
    # bar 2 ve 3'te iki etiket birden açık.
    assert co.iloc[2] == 2 and co.iloc[3] == 2
    assert co.iloc[1] == 1 and co.iloc[7] == 1
    uniq = get_average_uniqueness(idx, t1, co)
    assert 0 < uniq.min() <= uniq.max() <= 1.0


def test_cusum_reduces_event_count() -> None:
    """CUSUM filtresi bar sayısından çok daha az olay üretmeli."""
    df = _sample_data()
    tv = get_target_volatility(df["close"], cfg=_CFG.labeling)
    ev = cusum_filter(df["close"], tv * _CFG.labeling.cusum_threshold_mult)
    assert 0 < len(ev) < len(df) / 5


# --------------------------------------------------------------------------- #
# SIZINTI: çapraz doğrulama
# --------------------------------------------------------------------------- #


def test_purged_kfold_removes_overlapping_train_samples() -> None:
    """Purged K-Fold, test etiketleriyle çakışan eğitim örneği BIRAKMAMALI."""
    df = _sample_data()
    events = _events(df)
    t1 = events["t1"]
    cv = PurgedKFold(t1=t1, n_splits=4, embargo_pct=0.02)

    t0_all = t1.index
    t1_all = pd.DatetimeIndex(t1.to_numpy())

    for tr, te in cv.split(pd.DataFrame(index=t1.index)):
        assert len(np.intersect1d(tr, te)) == 0, "Eğitim ve test kesişiyor."
        test_start = t0_all[te.min()]
        test_end = t1_all[te].max()
        overlap = (t1_all[tr] >= test_start) & (t0_all[tr] <= test_end)
        assert not overlap.any(), f"{overlap.sum()} çakışan eğitim örneği purge edilmemiş."


def test_embargo_creates_gap_after_test_block() -> None:
    """Embargo, test bloğunun hemen ardındaki örnekleri atmalı."""
    n = 400
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    # Çakışmayan etiketler: purging etkisiz, saf embargo görünür.
    t1 = pd.Series(idx, index=idx)
    cv = PurgedKFold(t1=t1, n_splits=4, embargo_pct=0.05)
    embargo = int(n * 0.05)
    for tr, te in cv.split(pd.DataFrame(index=idx)):
        after = np.arange(te.max() + 1, min(n, te.max() + 1 + embargo))
        assert len(np.intersect1d(tr, after)) == 0, "Embargo bölgesi eğitimde kalmış."


def test_expected_max_sharpe_grows_with_trials() -> None:
    """Daha çok deneme, daha yüksek 'şans eşiği' anlamına gelmeli."""
    a = expected_max_sharpe(10, 0.04)
    b = expected_max_sharpe(1_000, 0.04)
    assert 0 < a < b


def test_deflated_sharpe_penalises_many_trials() -> None:
    """Aynı Sharpe, deneme sayısı arttıkça daha düşük DSR vermeli."""
    few = deflated_sharpe_ratio(0.08, 1_000, 5, 0.004)
    many = deflated_sharpe_ratio(0.08, 1_000, 5_000, 0.004)
    assert few["dsr"] > many["dsr"]


# --------------------------------------------------------------------------- #
# Boyutlandırma
# --------------------------------------------------------------------------- #


def test_bet_size_monotone_in_probability() -> None:
    """Olasılık arttıkça bahis büyüklüğü artmalı, p=0.5'te sıfır olmalı."""
    p = np.array([0.5, 0.55, 0.6, 0.8, 0.95])
    s = prob_to_bet_size(p)
    assert abs(s[0]) < 1e-9
    assert np.all(np.diff(s) > 0)
    assert s.max() <= 1.0


def test_kelly_is_zero_below_even_odds() -> None:
    """Kazanma olasılığı 0.5'in altındayken Kelly sıfır (bahis yok) olmalı."""
    assert kelly_size(np.array([0.3, 0.5]), payoff_ratio=1.0).max() == 0.0
    assert kelly_size(np.array([0.9]), payoff_ratio=1.0, fraction=1.0, cap=1.0)[0] > 0


def test_position_size_respects_caps_and_direction() -> None:
    """Boyut kapakları ve yön korunmalı."""
    cfg = ResearchConfig().sizing
    size = compute_position_size(
        np.array([0.9, 0.9]), np.array([1.0, -1.0]), vol_forecast=np.array([1e-3, 1e-3]),
        payoff_ratio=1.0, cfg=cfg,
    )
    assert size[0] > 0 and size[1] < 0
    assert np.abs(size).max() <= cfg.max_position + 1e-9


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_model_roundtrip_preserves_predictions() -> None:
    """Kaydedilip yüklenen model aynı tahminleri üretmeli."""
    df = _sample_data()
    X, events, w = _dataset(df)
    m = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    p1 = m.predict_proba(X)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_model(m, Path(tmp) / "m.pkl")
        assert path.with_suffix(".meta.json").exists()
        m2 = load_model(path)
    p2 = m2.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=0, atol=0)
    assert m2.feature_names == m.feature_names
    assert m2.config, "Config modelle birlikte saklanmalı."


def test_feature_parity_check_catches_schema_drift() -> None:
    """Sütun şeması değişirse canlı taraf SESSİZCE devam etmemeli."""
    df = _sample_data()
    X, events, w = _dataset(df)
    m = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    check_feature_parity(m, X)  # aynı şema: sorun yok
    try:
        check_feature_parity(m, X.drop(columns=[X.columns[0]]))
    except ValueError:
        return
    raise AssertionError("Eksik sütun ValueError atmalıydı.")


def test_training_is_deterministic() -> None:
    """Aynı tohumla iki eğitim aynı tahminleri vermeli."""
    df = _sample_data()
    X, events, w = _dataset(df)
    a = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    b = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-12)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #


def test_costs_strictly_reduce_performance() -> None:
    """Maliyet arttıkça net getiri DÜŞMELİ — motorun temel akıl sağlığı testi."""
    import copy

    df = _sample_data()
    X, events, w = _dataset(df)
    m = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    prob = pd.Series(m.predict_proba(X), index=X.index)

    free = copy.deepcopy(_CFG.backtest)
    free.commission_rate = 0.0
    free.slippage_bps = 0.0
    free.min_edge_over_cost = 0.0

    costly = copy.deepcopy(free)
    costly.commission_rate = 0.002
    costly.slippage_bps = 5.0

    r_free = run_backtest(df, events, prob, cfg=free, sizing_cfg=_CFG.sizing,
                          label_cfg=_CFG.labeling, data_cfg=_CFG.data)
    r_cost = run_backtest(df, events, prob, cfg=costly, sizing_cfg=_CFG.sizing,
                          label_cfg=_CFG.labeling, data_cfg=_CFG.data)
    assert r_free.metrics["islem_sayisi"] > 0
    assert r_cost.metrics["toplam_getiri"] < r_free.metrics["toplam_getiri"]
    assert total_cost_rate(costly) > total_cost_rate(free)


def test_latency_delays_execution() -> None:
    """Emir, sinyal barında DEĞİL, gecikme kadar sonra gerçekleşmeli."""
    df = _sample_data()
    X, events, w = _dataset(df)
    m = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    prob = pd.Series(m.predict_proba(X), index=X.index)

    import copy

    cfg = copy.deepcopy(_CFG.backtest)
    cfg.min_edge_over_cost = 0.0
    cfg.prob_threshold = 0.5
    res = run_backtest(df, events, prob, cfg=cfg, sizing_cfg=_CFG.sizing,
                       label_cfg=_CFG.labeling, data_cfg=_CFG.data)
    if res.trades.empty:
        return
    lat = pd.Timedelta(minutes=cfg.latency_bars * _CFG.data.bar_minutes)
    assert (res.trades["entry_time"] - res.trades["event_time"] == lat).all()
    assert (res.trades["exit_time"] > res.trades["entry_time"]).all()


def test_no_overlapping_positions_by_default() -> None:
    """Varsayılan ayarda aynı anda birden fazla pozisyon açılmamalı."""
    df = _sample_data()
    X, events, w = _dataset(df)
    m = train_model(X, events["label"], sample_weight=w["weight"], cfg=_CFG.model, full_config=_CFG)
    prob = pd.Series(m.predict_proba(X), index=X.index)
    res = run_backtest(df, events, prob, cfg=_CFG.backtest, sizing_cfg=_CFG.sizing,
                       label_cfg=_CFG.labeling, data_cfg=_CFG.data)
    if len(res.trades) < 2:
        return
    entries = res.trades["entry_pos"].to_numpy()
    exits = res.trades["exit_pos"].to_numpy()
    assert (entries[1:] > exits[:-1]).all(), "Çakışan pozisyon açılmış."


# --------------------------------------------------------------------------- #
# Poll-forward toplayıcı
# --------------------------------------------------------------------------- #


def _make_ticks(
    prices: list[float],
    volumes: list[float],
    start: str = "2026-08-02 10:00:00",
    step_seconds: int = 5,
    spread: float = 100.0,
    symbol: str = "BTC_TL",
) -> pd.DataFrame:
    """Test için elle kurulmuş tick tablosu."""
    ts = pd.date_range(start, periods=len(prices), freq=f"{step_seconds}s", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "symbol": symbol,
            "last": prices,
            "lowest_ask": [p + spread / 2 for p in prices],
            "highest_bid": [p - spread / 2 for p in prices],
            "volume24h": volumes,
        }
    )


def test_ticker_parsing_handles_strings_and_missing_pairs() -> None:
    """Borsa sayıları dizge döndürebilir; olmayan parite çökmeye yol açmamalı."""
    from ..tools.collect_paribu import parse_ticker

    from ..tools.collect_paribu import _to_float

    payload = {
        "BTC_TL": {"last": "3000000.50", "lowestAsk": 3000100, "highestBid": "2999900",
                   "volume": "12.5"},
    }
    ts = pd.Timestamp("2026-08-02 10:00:00", tz="UTC")
    rows = parse_ticker(payload, ["BTC_TL", "YOK_TL"], ts)
    assert len(rows) == 1, "Olmayan parite atlanmalı, hata atmamalı."
    assert rows[0]["last"] == 3_000_000.50
    assert rows[0]["lowest_ask"] == 3_000_100.0, "Sayısal (dizge olmayan) değer de çalışmalı."
    assert rows[0]["volume24h"] == 12.5

    # EN KRİTİK: ondalıklı dizge asla binlik ayıracı sanılıp bozulmamalı.
    assert _to_float("3000.50") == 3000.50, "Ondalık nokta silinirse fiyat 100 katına çıkar!"
    # TR biçimi ancak standart çözüm BAŞARISIZ olursa devreye girer.
    assert _to_float("3.000.000,25") == 3_000_000.25
    assert _to_float("") != _to_float(""), "Boş değer NaN olmalı (NaN != NaN)."


def test_tick_storage_is_append_only_and_deduplicated() -> None:
    """Tekrar yazımlar veri kaybettirmemeli, çift kayıt da bırakmamalı."""
    import research.tools.collect_paribu as cp

    with tempfile.TemporaryDirectory() as tmp:
        orig = cp.CONFIG.collector.tick_dir
        cp.CONFIG.collector.tick_dir = tmp
        try:
            first = _make_ticks([100.0, 101.0], [10.0, 11.0])
            second = _make_ticks([102.0, 103.0], [12.0, 13.0], start="2026-08-02 10:00:10")
            cp.append_ticks(first.to_dict("records"))
            cp.append_ticks(second.to_dict("records"))
            cp.append_ticks(second.to_dict("records"))  # aynı veriyi tekrar yaz

            back = cp.load_ticks(symbol="BTC_TL")
            assert len(back) == 4, f"4 benzersiz tick beklenirdi, {len(back)} bulundu."
            assert back["ts"].is_monotonic_increasing
            assert not back["ts"].duplicated().any()
        finally:
            cp.CONFIG.collector.tick_dir = orig


def test_resample_builds_ohlc_from_last_prices() -> None:
    """OHLC 'last' tick'lerinden doğru kurulmalı."""
    from ..tools.collect_paribu import resample_ticks_to_ohlcv

    ticks = _make_ticks([100.0, 105.0, 95.0, 102.0], [10.0, 11.0, 12.0, 13.0])
    bars = resample_ticks_to_ohlcv(ticks, now=pd.Timestamp("2026-08-02 11:00:00", tz="UTC"))
    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["open"] == 100.0 and row["close"] == 102.0
    assert row["high"] == 105.0 and row["low"] == 95.0
    assert row["tick_count"] == 4
    assert bars.index.tz is not None and str(bars.index.tz) == "UTC"


def test_resample_drops_unclosed_bar() -> None:
    """Kapanmamış son bar atılmalı (README bölüm 10 sözleşmesi)."""
    from ..tools.collect_paribu import resample_ticks_to_ohlcv

    # 10:00 ve 10:01 dakikalarına yayılan tick'ler.
    ticks = _make_ticks([100.0] * 24, list(np.arange(24.0)), step_seconds=5)
    now = pd.Timestamp("2026-08-02 10:01:30", tz="UTC")  # 10:01 barı henüz kapanmadı
    kept = resample_ticks_to_ohlcv(ticks, drop_unclosed=True, now=now)
    all_bars = resample_ticks_to_ohlcv(ticks, drop_unclosed=False, now=now)
    assert len(all_bars) == 2
    assert len(kept) == 1, "Kapanmamış bar atılmalıydı."
    assert kept.index[-1] == pd.Timestamp("2026-08-02 10:00:00", tz="UTC")


def test_resample_volume_excludes_unmeasurable_intervals() -> None:
    """24s kümülatif hacimdeki negatif sıçrama ölçülemez sayılmalı, uydurulmamalı."""
    from ..tools.collect_paribu import resample_ticks_to_ohlcv

    # 100 -> 110 -> 120 -> 5 (gün dönümü sıfırlanması) -> 15
    ticks = _make_ticks([100.0] * 5, [100.0, 110.0, 120.0, 5.0, 15.0])
    bars = resample_ticks_to_ohlcv(ticks, now=pd.Timestamp("2026-08-02 11:00:00", tz="UTC"))
    row = bars.iloc[0]
    # Geçerli farklar: +10, +10, +10  (negatif olan -115 DIŞLANIR)
    assert row["volume"] == 30.0, f"Beklenen 30.0, bulunan {row['volume']}"
    assert abs(row["volume_gecerli_oran"] - 0.75) < 1e-9, "4 farkın 3'ü geçerli olmalı."


def test_volume_diagnosis_distinguishes_rolling_from_daily_reset() -> None:
    """Tanılama, kayan 24s penceresini gün sonu sıfırlanmasından ayırmalı."""
    from ..tools.collect_paribu import diagnose_volume_series

    # Gün ortasına yayılmış negatif farklar -> kayan pencere.
    rolling = _make_ticks([100.0] * 6, [100.0, 99.0, 101.0, 100.0, 102.0, 101.0],
                          start="2026-08-02 12:00:00")
    d1 = diagnose_volume_series(rolling)
    assert d1["mod"] == "kayan_24s", d1

    # Tek negatif fark, tam gün dönümünde -> günlük sıfırlanan sayaç.
    reset = _make_ticks([100.0] * 4, [100.0, 110.0, 5.0, 15.0], start="2026-08-02 23:59:50")
    d2 = diagnose_volume_series(reset)
    assert d2["mod"] == "gunluk_sifirlanan", d2


def test_resample_writes_spread_columns() -> None:
    """Spread sütunları mikroyapı sinyali olarak yazılmalı."""
    from ..tools.collect_paribu import resample_ticks_to_ohlcv

    ticks = _make_ticks([1000.0, 1000.0], [10.0, 11.0], spread=10.0)
    bars = resample_ticks_to_ohlcv(ticks, now=pd.Timestamp("2026-08-02 11:00:00", tz="UTC"))
    row = bars.iloc[0]
    assert abs(row["spread_mean"] - 10.0) < 1e-9
    assert abs(row["spread_rel_mean"] - 0.01) < 1e-9, "10/1000 = %1"


# --------------------------------------------------------------------------- #
# Bar toplulaştırma deneyi (--resample)
# --------------------------------------------------------------------------- #


def test_resample_ohlcv_aggregation_is_correct() -> None:
    """open=ilk, high=maks, low=min, close=son, volume=toplam olmalı."""
    from ..data import resample_ohlcv

    idx = pd.date_range("2026-01-01 00:00", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(100.0, 130.0),
            "high": np.arange(100.0, 130.0) + 2.0,
            "low": np.arange(100.0, 130.0) - 2.0,
            "close": np.arange(100.0, 130.0) + 0.5,
            "volume": np.ones(30),
        },
        index=idx,
    )
    out = resample_ohlcv(df, "15min", now=pd.Timestamp("2026-01-01 02:00", tz="UTC"))
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 100.0, "open ilk barin acilisi olmali."
    assert first["high"] == df["high"].iloc[:15].max()
    assert first["low"] == df["low"].iloc[:15].min()
    assert first["close"] == df["close"].iloc[14], "close son barin kapanisi olmali."
    assert first["volume"] == 15.0, "volume toplam olmali."
    assert out.index[0] == idx[0], "Barlar ACILIS zamaniyla etiketlenmeli (sag etiket = look-ahead)."


def test_resample_drops_incomplete_tail_bar() -> None:
    """Kaynak barları eksik olan son aralık atılmalı (yarım bar tam sayılmamalı)."""
    from ..data import resample_ohlcv

    idx = pd.date_range("2026-01-01 00:00", periods=23, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0},
        index=idx,
    )
    now = pd.Timestamp("2026-01-01 05:00", tz="UTC")  # her sey duvar saatine gore kapali
    out = resample_ohlcv(df, "15min", drop_incomplete_tail=True, now=now)
    kept = resample_ohlcv(df, "15min", drop_incomplete_tail=False, now=now)
    assert len(kept) == 2, "00:00 ve 00:15 araliklari"
    assert len(out) == 1, "00:15 araliginda yalnizca 8 dakika var -> atilmali."


def test_scale_config_keeps_wall_clock_horizon() -> None:
    """Tutma ufku bar boyutundan bağımsız olarak aynı SÜREYİ vermeli."""
    from ..config import ResearchConfig, scale_config_for_bars

    base = ResearchConfig()  # 1dk bar, 240 bar ufuk = 4 saat
    base_hours = base.labeling.vertical_bars * base.data.bar_minutes / 60.0

    for minutes, beklenen_bar in ((15, 16), (60, 4)):
        scaled = scale_config_for_bars(base, minutes)
        assert scaled.data.bar_minutes == minutes
        assert scaled.labeling.vertical_bars == beklenen_bar
        hours = scaled.labeling.vertical_bars * scaled.data.bar_minutes / 60.0
        assert abs(hours - base_hours) < 1e-9, f"{minutes}dk: ufuk {hours}s, beklenen {base_hours}s"

    # Girdi degistirilmemeli (yan etki yok).
    assert base.data.bar_minutes == 1 and base.labeling.vertical_bars == 240


# --------------------------------------------------------------------------- #
# Binance geçmiş veri indirici
# --------------------------------------------------------------------------- #


def _kline_rows(start_ms: int, n: int, unit_mult: int = 1, price: float = 100.0) -> list[list]:
    """Binance kline satırları üretir (REST dizisi ve CSV satırı aynı düzendedir).

    Args:
        start_ms: İlk barın açılış zamanı (ms).
        n: Bar sayısı.
        unit_mult: Zaman damgası çarpanı (1=ms, 1000=mikrosaniye dökümleri).
        price: Baz fiyat.
    """
    rows = []
    for i in range(n):
        t = (start_ms + i * 60_000) * unit_mult
        p = price + i * 0.5
        rows.append([
            t, f"{p:.2f}", f"{p + 1:.2f}", f"{p - 1:.2f}", f"{p + 0.25:.2f}", "12.5",
            t + 59_999 * unit_mult, "1250.0", 42, "6.0", "600.0", "0",
        ])
    return rows


def test_binance_epoch_unit_is_measured_not_assumed() -> None:
    """Zaman damgası birimi büyüklükten çıkarılmalı (ms/µs karışırsa veri çöp olur)."""
    from ..tools.fetch_binance import infer_epoch_unit

    assert infer_epoch_unit(1_754_000_000) == "s"
    assert infer_epoch_unit(1_754_000_000_000) == "ms"
    assert infer_epoch_unit(1_754_000_000_000_000) == "us"


def test_binance_zip_parsing_handles_header_and_microseconds() -> None:
    """Toplu döküm zip'i: başlık satırı ve mikrosaniye damgası doğru işlenmeli."""
    import csv
    import io as _io
    import zipfile as _zip

    from ..tools.fetch_binance import parse_kline_zip

    start_ms = int(pd.Timestamp("2026-06-01 00:00:00", tz="UTC").timestamp() * 1000)
    rows = _kline_rows(start_ms, 5, unit_mult=1000)  # mikrosaniye dökümü

    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(range(12)))  # yeni dökümlerdeki başlık satırı (sayısal değil)
    buf.seek(0)
    text = "open_time,open,high,low,close,volume,close_time,q,n,tb,tq,ig\n"
    text += "\n".join(",".join(str(c) for c in r) for r in rows)

    zbuf = _io.BytesIO()
    with _zip.ZipFile(zbuf, "w") as zf:
        zf.writestr("BTCTRY-1m-2026-06.csv", text)

    df = parse_kline_zip(zbuf.getvalue())
    assert len(df) == 5, "Başlık satırı atılmalı, 5 bar kalmalı."
    assert df.index[0] == pd.Timestamp("2026-06-01 00:00:00", tz="UTC"), "µs birimi çözülmeli."
    assert df.index[1] - df.index[0] == pd.Timedelta(minutes=1)
    assert df["close"].iloc[0] == 100.25


def test_binance_rest_rows_parse_to_contract() -> None:
    """REST yanıtı (dizi dizisi) doğru çerçeveye dönmeli."""
    from ..tools.fetch_binance import klines_to_frame

    start_ms = int(pd.Timestamp("2026-06-01 00:00:00", tz="UTC").timestamp() * 1000)
    df = klines_to_frame(_kline_rows(start_ms, 3))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df["volume"].iloc[0] == 12.5


def test_binance_normalize_enforces_pipeline_contract() -> None:
    """Sıralama, tekilleştirme ve kapanmamış bar atma uygulanmalı."""
    from ..tools.fetch_binance import klines_to_frame, normalize

    start_ms = int(pd.Timestamp("2026-06-01 00:00:00", tz="UTC").timestamp() * 1000)
    df = klines_to_frame(_kline_rows(start_ms, 10))
    shuffled = pd.concat([df.iloc[5:], df.iloc[:5], df.iloc[:2]])  # karışık + tekrarlı

    now = pd.Timestamp("2026-06-01 00:09:30", tz="UTC")  # son bar (00:09) kapanmadı
    out = normalize(shuffled, now=now)
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert len(out) == 9, "Kapanmamış son bar atılmalıydı."
    assert out.index[-1] == pd.Timestamp("2026-06-01 00:08:00", tz="UTC")


def test_binance_symbol_choice_measures_coverage_and_falls_back() -> None:
    """BTCTRY ince ise otomatik BTCUSDT'ye düşmeli — varsayımla değil, ÖLÇÜMLE."""
    from ..tools.fetch_binance import choose_symbol, klines_to_frame

    now = datetime(2026, 6, 3, tzinfo=timezone.utc)
    start_ms = int((now - timedelta(days=2)).timestamp() * 1000)

    def thin_try(symbol, a, b):
        # BTCTRY: 2 gunluk pencerede sadece 100 bar -> %3.5 doluluk (ince).
        n = 100 if symbol == "BTCTRY" else 2880
        return klines_to_frame(_kline_rows(start_ms, n)), {}

    chosen, reports = choose_symbol(downloader=thin_try, now=now)
    assert chosen == "BTCUSDT", f"Ince BTCTRY yerine BTCUSDT secilmeliydi, secilen: {chosen}"
    assert reports[0]["symbol"] == "BTCTRY" and not reports[0]["kullanilabilir"]
    assert reports[0]["doluluk"] < 0.1

    def both_full(symbol, a, b):
        return klines_to_frame(_kline_rows(start_ms, 2880)), {}

    chosen2, _ = choose_symbol(downloader=both_full, now=now)
    assert chosen2 == "BTCTRY", "Yeterli veri varsa tercih sirasi korunmali (once TL)."


def test_binance_bulk_urls_split_monthly_and_daily() -> None:
    """Tamamlanmış aylar için aylık, içinde bulunulan ay için günlük zip kullanılmalı."""
    from ..tools.fetch_binance import bulk_urls

    urls = bulk_urls(
        "BTCUSDT",
        datetime(2026, 5, 15, tzinfo=timezone.utc),
        datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    monthly = [u for u in urls if "/monthly/" in u]
    daily = [u for u in urls if "/daily/" in u]
    assert len(monthly) == 3, f"2026-05/06/07 beklenirdi: {monthly}"
    assert "BTCUSDT-1m-2026-07.zip" in monthly[-1]
    assert len(daily) == 2, f"1-2 Agustos beklenirdi: {daily}"
    assert daily[0].endswith("BTCUSDT-1m-2026-08-01.zip")


def test_binance_download_feeds_the_pipeline() -> None:
    """İndirilen veri doğrudan make_features'a girebilmeli (şema uyumu)."""
    from ..tools.fetch_binance import fetch_history, klines_to_frame

    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    start_ms = int((now - timedelta(days=3)).timestamp() * 1000)

    def fake(symbol, a, b):
        return klines_to_frame(_kline_rows(start_ms, 3 * 1440, price=3_000_000.0)), {"kaynak": "test"}

    with tempfile.TemporaryDirectory() as tmp:
        import research.tools.fetch_binance as fb

        orig = fb.download_rest
        fb.download_rest = fake  # type: ignore[assignment]
        try:
            bars, meta = fetch_history(symbol="BTCUSDT", days=3, source="rest",
                                       out_dir=tmp, now=now)
        finally:
            fb.download_rest = orig  # type: ignore[assignment]

    assert len(bars) > 4000, f"3 gunluk 1m veri beklenirdi, {len(bars)} geldi."
    assert meta["secilen_sembol"] == "BTCUSDT"
    assert Path(meta["dosya"]).name == "BTCUSDT_1m_binance.parquet", "Sembol dosya adinda olmali."
    assert list(bars.columns) == list(OHLCV_COLUMNS)
    X = make_features(bars)
    assert X.shape[0] == len(bars) and X.shape[1] > 50
    assert X["mk_spread_rel"].isna().all(), "Binance'te defter yok; mk_* NaN kalmali."


# --------------------------------------------------------------------------- #
# Grid stratejisi
# --------------------------------------------------------------------------- #


def _price_path(prices: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Verilen kapanış dizisinden OHLCV çerçevesi (high/low = kapanışları kapsar)."""
    idx = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    p = np.asarray(prices, dtype="float64")
    prev = np.concatenate([[p[0]], p[:-1]])
    return pd.DataFrame(
        {
            "open": prev,
            "high": np.maximum(prev, p),
            "low": np.minimum(prev, p),
            "close": p,
            "volume": np.ones(len(p)),
        },
        index=idx,
    )


def test_grid_levels_spacing() -> None:
    """Geometrik kademelerde her adım eşit YÜZDE olmalı."""
    from ..grid_backtest import build_levels, grid_step_pct

    lv = build_levels(100.0, 200.0, 11, "geometric")
    assert len(lv) == 11 and lv[0] == 100.0 and lv[-1] == 200.0
    steps = np.diff(lv) / lv[:-1]
    assert np.allclose(steps, steps[0]), "Geometrik aralikta adimlar esit yuzde olmali."
    assert abs(grid_step_pct(lv) - steps[0]) < 1e-12

    lin = build_levels(100.0, 200.0, 11, "linear")
    assert np.allclose(np.diff(lin), 10.0)


def test_grid_buys_on_drop_and_sells_on_rise() -> None:
    """Bir kademe düşünce al, bir kademe çıkınca sat."""
    from ..config import GridConfig
    from ..grid_backtest import run_grid_backtest

    cfg = GridConfig(lower_price=90.0, upper_price=110.0, n_levels=11,
                     total_capital=11_000.0, spacing="linear",
                     commission_rate=0.0, slippage_bps=0.0)
    # 100'den 96'ya in (alislar), sonra 104'e cik (satislar).
    df = _price_path([100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 103, 104])
    res = run_grid_backtest(df, cfg)

    assert not res.trades.empty, "Hic islem olusmadi."
    buys = res.trades[res.trades["yon"] == "AL"]
    sells = res.trades[res.trades["yon"] == "SAT"]
    assert len(buys) >= 4, f"Dususte alim bekleniyordu: {len(buys)}"
    assert len(sells) >= 4, f"Yukseliste satim bekleniyordu: {len(sells)}"
    assert (res.pairs["kar"] > 0).all(), "Maliyetsiz gridde her cift kar etmeli."
    # Her cift bir kademe (=2 birim fiyat) kar etmeli.
    assert res.metrics["gerceklesmis_kar"] > 0


def test_grid_costs_are_applied_to_every_trade() -> None:
    """Maliyet her işleme uygulanmalı; maliyetli sonuç maliyetsizden kötü olmalı."""
    from ..config import GridConfig
    from ..grid_backtest import run_grid_backtest

    df = _price_path([100, 98, 96, 98, 100, 98, 96, 98, 100])
    free = GridConfig(lower_price=90.0, upper_price=110.0, n_levels=11,
                      total_capital=11_000.0, spacing="linear",
                      commission_rate=0.0, slippage_bps=0.0)
    costly = GridConfig(lower_price=90.0, upper_price=110.0, n_levels=11,
                        total_capital=11_000.0, spacing="linear",
                        commission_rate=0.002, slippage_bps=5.0)

    r_free = run_grid_backtest(df, free)
    r_cost = run_grid_backtest(df, costly)

    assert r_free.metrics["toplam_maliyet"] == 0.0
    assert r_cost.metrics["toplam_maliyet"] > 0.0
    assert r_cost.metrics["toplam_kz"] < r_free.metrics["toplam_kz"], \
        "Maliyet toplam sonucu KOTULESTIRMELI."
    assert r_cost.metrics["gidis_donus_maliyet"] == 2 * (0.002 + 0.0005)


def test_grid_unrealized_loss_is_reported_separately() -> None:
    """EN KRİTİK: fiyat aralıktan düşünce elde kalan coin'in zararı ayrı görünmeli."""
    from ..config import GridConfig
    from ..grid_backtest import run_grid_backtest

    cfg = GridConfig(lower_price=90.0, upper_price=110.0, n_levels=11,
                     total_capital=11_000.0, spacing="linear",
                     commission_rate=0.0, slippage_bps=0.0)
    # Once biraz zikzak (gerceklesmis kar olussun), sonra sert dusus.
    path = [100, 98, 100, 98, 96, 94, 92, 90, 85, 80, 75, 70]
    res = run_grid_backtest(_price_path(path), cfg)
    m = res.metrics

    assert m["elde_kalan_miktar"] > 0, "Dususte elde envanter kalmali."
    assert m["gerceklesmemis_kz"] < 0, "Elde kalan coin zararda olmali."
    assert m["aralik_kirildi"] is True, "Fiyat alt sinirin altina indi."
    assert m["alt_sinir_alti_bar"] > 0
    assert m["aralik_disi_sure_orani"] > 0

    # Toplam = gerceklesmis + gerceklesmemis (muhasebe tutarliligi).
    assert abs(m["toplam_kz"] - (m["gerceklesmis_kar"] + m["gerceklesmemis_kz"])) < 1e-6
    # Sermaye egrisi de ayni sonuca varmali.
    assert abs(res.equity.iloc[-1] - (cfg.total_capital + m["toplam_kz"])) < 1e-6
    # Ve asil mesele: gerceklesmis kar pozitifken toplam negatif olabilmeli.
    assert m["gerceklesmis_kar"] > 0 and m["toplam_kz"] < 0, \
        "Bu senaryo tam da 'gizlenen risk' vakasi olmali."


def test_grid_never_spends_money_it_does_not_have() -> None:
    """Nakit asla negatife düşmemeli; yetmezse alım atlanmalı (bedava kaldıraç yok).

    Not: statik gridde sermaye kademelere ÖNCEDEN bölündüğü için (kademe başına
    ``sermaye / n_levels``) alımların toplamı tanım gereği sermayeyi aşamaz.
    Nakit tükenmesi ancak MALİYETLER eklendiğinde (veya trailing modda grid
    kayıp yeniden alım yapıldığında) ortaya çıkar. Test bu mekanizmayı yüksek
    komisyonla zorlar.
    """
    from ..config import GridConfig
    from ..grid_backtest import run_grid_backtest

    düsüs = _price_path(list(range(100, 49, -1)))
    base = dict(lower_price=50.0, upper_price=110.0, n_levels=21,
                total_capital=200.0, spacing="linear", slippage_bps=0.0)

    # 1) Maliyetsiz: onceden bolunmus sermaye tasmaz, atlama olmaz.
    r0 = run_grid_backtest(düsüs, GridConfig(**base, commission_rate=0.0))
    assert r0.cash.min() >= -1e-9, "Nakit negatife dusmemeli."
    assert r0.metrics["atlanan_alis_nakit_yok"] == 0

    # 2) Asiri maliyet (%30 — gercekci degil, korumayi tetiklemek icin): son
    #    alimlara para kalmaz, atlanmali.
    r1 = run_grid_backtest(düsüs, GridConfig(**base, commission_rate=0.30))
    assert r1.cash.min() >= -1e-9, "Nakit maliyetle bile negatife dusmemeli."
    assert r1.metrics["atlanan_alis_nakit_yok"] > 0, "Maliyet nakiti tuketmeliydi."


def test_grid_regime_analysis_labels_trends() -> None:
    """Alt dönemler rejim etiketiyle raporlanmalı."""
    from ..config import GridConfig
    from ..grid_backtest import label_trend, regime_analysis, run_grid_backtest

    assert label_trend(100, 110, 0.02) == "yukselis"
    assert label_trend(100, 90, 0.02) == "dusus"
    assert label_trend(100, 100.5, 0.02) == "yatay"

    cfg = GridConfig(lower_price=90.0, upper_price=110.0, n_levels=11,
                     total_capital=11_000.0, spacing="linear")
    n = 24 * 21  # 3 hafta saatlik
    prices = 100 + 3 * np.sin(np.arange(n) / 12.0)
    df = _price_path(list(prices))
    res = run_grid_backtest(df, cfg)
    reg = regime_analysis(df, res.equity, cfg)
    assert len(reg) >= 2, "Haftalik pencereler olusmali."
    assert set(reg["rejim"]).issubset({"yukselis", "dusus", "yatay"})
    assert {"strateji_getiri", "strateji_max_dd", "fiyat_degisim"} <= set(reg.columns)


def test_grid_trailing_mode_shifts_range_up() -> None:
    """Hareketli modda fiyat üst sınırı aşınca grid yukarı kaymalı."""
    from ..config import GridConfig
    from ..grid_backtest import run_grid_backtest

    base = dict(lower_price=90.0, upper_price=110.0, n_levels=11,
                total_capital=11_000.0, spacing="linear",
                commission_rate=0.0, slippage_bps=0.0)
    df = _price_path(list(range(100, 141)))  # surekli yukselis

    static = run_grid_backtest(df, GridConfig(**base, mode="static"))
    trailing = run_grid_backtest(df, GridConfig(**base, mode="trailing"))

    assert static.metrics["grid_kaydirma"] == 0
    assert trailing.metrics["grid_kaydirma"] > 0, "Trailing modda kaydirma olmaliydi."
    assert trailing.levels[-1] > static.levels[-1], "Aralik yukari tasinmali."
    assert static.metrics["ust_sinir_ustu_bar"] > 0, "Sabit grid aralik disinda kalmali."


# --------------------------------------------------------------------------- #
# Üçgen arbitraj ölçüm aracı
# --------------------------------------------------------------------------- #


def _ticker(pairs: dict[str, tuple[float, float]]) -> dict:
    """``{"BTC_TL": (ask, bid)}`` -> Paribu ticker biçimi."""
    return {
        sym: {"last": (a + b) / 2, "lowestAsk": a, "highestBid": b, "volume": "1"}
        for sym, (a, b) in pairs.items()
    }


def test_triangular_universe_detects_missing_coin_coin_pairs() -> None:
    """Sadece TL pariteleri varsa üçgen YAPISAL olarak imkânsız — bu raporlanmalı."""
    from ..tools.triangular_paribu import describe_universe

    only_tl = _ticker({"BTC_TL": (100.0, 99.0), "ETH_TL": (10.0, 9.9),
                       "USDT_TL": (34.0, 33.9)})
    u = describe_universe(only_tl)
    assert u["coin_coin"] == [], "TL disi parite yok."
    assert u["ucgen_mumkun"] is False, "Ucgen imkansiz olarak raporlanmali."

    with_cross = dict(only_tl, **_ticker({"BTC_USDT": (3.0, 2.9)}))
    u2 = describe_universe(with_cross)
    assert u2["coin_coin"] == ["BTC_USDT"]
    assert u2["ucgen_mumkun"] is True
    assert set(u2["para_birimleri"]) == {"BTC", "ETH", "USDT", "TL"}


def test_triangular_uses_ask_when_buying_and_bid_when_selling() -> None:
    """GERÇEKÇİ yön: alışta lowestAsk ödenmeli, satışta highestBid alınmalı."""
    from ..tools.triangular_paribu import build_markets, convert

    markets = build_markets(_ticker({"BTC_TL": (110.0, 90.0)}))

    # TL -> BTC: BTC ALIYORUZ, ask (110) odemeliyiz.
    out, leg = convert(1100.0, "TL", "BTC", markets, fee=0.0)
    assert leg["yon"] == "AL" and leg["fiyat"] == 110.0
    assert abs(out - 10.0) < 1e-12, "1100 TL / 110 ask = 10 BTC"

    # BTC -> TL: BTC SATIYORUZ, bid (90) almaliyiz.
    out2, leg2 = convert(10.0, "BTC", "TL", markets, fee=0.0)
    assert leg2["yon"] == "SAT" and leg2["fiyat"] == 90.0
    assert abs(out2 - 900.0) < 1e-12, "10 BTC * 90 bid = 900 TL"

    # Ortalama fiyat kullanilsaydi ikisi de 100 olur, spread kaybi gorunmezdi.
    assert out2 < 1100.0, "Spread her iki bacakta da aleyhe calismali."


def test_triangular_deducts_three_commissions() -> None:
    """Net getiri brütten TAM 3 işlem komisyonu kadar düşük olmalı."""
    from ..tools.triangular_paribu import build_markets, evaluate_cycle

    # Spread'siz, tam tutarli fiyatlar -> brut getiri tam 0.
    markets = build_markets(_ticker({
        "BTC_TL": (100.0, 100.0),
        "USDT_TL": (10.0, 10.0),
        "BTC_USDT": (10.0, 10.0),
    }))
    fee = 0.002
    r = evaluate_cycle(("TL", "BTC", "USDT"), markets, fee)
    assert r is not None
    assert abs(r["brut_getiri"]) < 1e-12, "Tutarli fiyatlarda brut getiri 0 olmali."
    assert abs(r["net_getiri"] - ((1 - fee) ** 3 - 1)) < 1e-12
    assert abs(r["toplam_komisyon"] - 3 * fee) < 1e-15
    assert r["kar_var"] is False
    assert len(r["bacaklar"]) == 3


def test_triangular_reports_no_opportunity_honestly() -> None:
    """Brüt pozitif ama komisyon sonrası negatifse 'fırsat var' DENMEMELİ."""
    from ..tools.triangular_paribu import measure

    # BTC_USDT hafif yanlis fiyatli: brut kucuk pozitif, ama %0.6 komisyonun alti.
    payload = _ticker({
        "BTC_TL": (100.0, 99.98),
        "USDT_TL": (10.0, 9.998),
        "BTC_USDT": (10.01, 10.008),
    })
    m = measure(payload, fee=0.002, start="TL")
    assert m["dongu_sayisi"] > 0
    best = m["sonuclar"][0]
    assert best["net_getiri"] < 0, "Komisyon brut kazanci yutmali."
    assert m["net_pozitif"] == [], "Net pozitif dongu YOK olarak raporlanmali."


def test_triangular_finds_genuine_opportunity_when_it_exists() -> None:
    """Gerçekten kârlı bir tutarsızlık varsa araç onu bulabilmeli."""
    from ..tools.triangular_paribu import measure

    # BTC_USDT belirgin sekilde ucuz -> TL->BTC->USDT->TL karli olmali.
    payload = _ticker({
        "BTC_TL": (100.0, 99.9),
        "USDT_TL": (10.0, 9.99),
        "BTC_USDT": (9.0, 8.99),   # BTC, USDT cinsinden cok ucuz
    })
    m = measure(payload, fee=0.002)
    assert len(m["net_pozitif"]) > 0, "Gercek firsat bulunamadi."
    assert m["sonuclar"][0]["net_getiri"] > 0


def test_triangular_watch_counts_opportunities_without_network() -> None:
    """--watch modu ağsız çalışmalı ve fırsatlı ölçümleri saymalı."""
    from ..tools.triangular_paribu import watch

    karli = _ticker({"BTC_TL": (100.0, 99.9), "USDT_TL": (10.0, 9.99),
                     "BTC_USDT": (9.0, 8.99)})
    karsiz = _ticker({"BTC_TL": (100.0, 99.9), "USDT_TL": (10.0, 9.99),
                      "BTC_USDT": (10.0, 9.99)})
    seq = [karli, karsiz, karsiz, karli]
    calls = {"i": 0}

    def getter() -> dict:
        p = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        return p

    ticks = {"t": 0.0}

    def clock() -> float:
        return ticks["t"]

    def sleeper(s: float) -> None:
        ticks["t"] += s

    w = watch(duration=4.0, interval=1.0, fee=0.002, getter=getter,
              sleeper=sleeper, clock=clock)
    assert w["ornek"] == 4 and w["hata"] == 0
    assert w["firsatli_ornek"] == 2, f"2 firsatli olcum beklenirdi: {w['firsatli_ornek']}"
    assert abs(w["firsat_orani"] - 0.5) < 1e-9
    assert w["dongu_bazinda"], "Dongu bazinda ozet uretilmeli."


def test_triangular_survives_broken_ticker() -> None:
    """Bozuk/eksik parite verisi çökmeye yol açmamalı."""
    from ..tools.triangular_paribu import build_markets, measure

    broken = {
        "BTC_TL": {"lowestAsk": "0", "highestBid": "abc"},   # gecersiz fiyat
        "GARIP": {"lowestAsk": "1", "highestBid": "1"},       # ayrisamayan sembol
        "ETH_TL": {"lowestAsk": "10", "highestBid": "9.9"},
        "BOS_TL": {},
    }
    markets = build_markets(broken)
    assert ("ETH", "TL") in markets and ("BTC", "TL") not in markets
    m = measure(broken, fee=0.002)
    assert m["dongu_sayisi"] == 0, "Ucgen kurulamamali ama cokme de olmamali."


# --------------------------------------------------------------------------- #
# Borsalar arası TL primi
# --------------------------------------------------------------------------- #


def test_cross_exchange_premium_math() -> None:
    """Prim formülü: (paribu / global - 1) * 100."""
    from ..tools.cross_exchange import compute_premium

    assert abs(compute_premium(105.0, 100.0) - 5.0) < 1e-12
    assert abs(compute_premium(97.0, 100.0) - (-3.0)) < 1e-12
    assert compute_premium(0.0, 100.0) != compute_premium(0.0, 100.0)  # NaN
    assert compute_premium(100.0, 0.0) != compute_premium(100.0, 0.0)  # NaN


def test_cross_exchange_prefers_direct_pair_then_falls_back_to_product() -> None:
    """BTCTRY varsa doğrudan; yoksa BTCUSDT × USDTTRY kullanılmalı."""
    from ..tools.cross_exchange import global_btc_try

    price, detail = global_btc_try(lambda s: {"BTCTRY": 3_500_000.0}.get(s))
    assert price == 3_500_000.0 and detail["yol"] == "dogrudan"

    price2, detail2 = global_btc_try(
        lambda s: {"BTCUSDT": 100_000.0, "USDTTRY": 34.0}.get(s)
    )
    assert abs(price2 - 3_400_000.0) < 1e-6 and detail2["yol"] == "carpim"

    price3, detail3 = global_btc_try(lambda s: None)
    assert price3 != price3 and detail3["yol"] == "yok", "Hicbiri yoksa NaN."


def test_cross_exchange_measure_once_without_network() -> None:
    """Enjekte edilmiş kaynaklarla ağsız ölçüm yapılabilmeli."""
    from ..tools.cross_exchange import PriceSources, measure_once

    src = PriceSources(
        paribu=lambda: _ticker({"BTC_TL": (3_570_000.0, 3_569_000.0)}),
        binance=lambda s: {"BTCUSDT": 100_000.0, "USDTTRY": 35.0}.get(s),
    )
    m = measure_once(src)
    assert m["gecerli"] is True
    assert abs(m["global"] - 3_500_000.0) < 1e-6
    # paribu 'last' = (ask+bid)/2 = 3_569_500
    assert abs(m["prim_yuzde"] - ((3_569_500 / 3_500_000 - 1) * 100)) < 1e-9
    assert m["yol"] == "carpim"


def test_cross_exchange_watch_summarises_distribution() -> None:
    """--watch primin ortalama/std/min/maks dağılımını vermeli."""
    from ..tools.cross_exchange import PriceSources, watch

    primler = [3_500_000.0, 3_535_000.0, 3_570_000.0, 3_605_000.0]  # %0, 1, 2, 3
    calls = {"i": 0}

    def paribu() -> dict:
        px = primler[min(calls["i"], len(primler) - 1)]
        calls["i"] += 1
        return _ticker({"BTC_TL": (px, px)})

    ticks = {"t": 0.0}
    src = PriceSources(paribu=paribu,
                       binance=lambda s: {"BTCUSDT": 100_000.0, "USDTTRY": 35.0}.get(s))

    w = watch(4.0, 1.0, sources=src, sleeper=lambda s: ticks.__setitem__("t", ticks["t"] + s),
              clock=lambda: ticks["t"], verbose=False)
    s = w["ozet"]
    assert s["n"] == 4 and w["hata"] == 0
    assert abs(s["min"] - 0.0) < 1e-6
    assert abs(s["maks"] - 3.0) < 1e-6
    assert abs(s["ortalama"] - 1.5) < 1e-6
    assert s["std"] > 0 and s["pozitif_oran"] == 0.75


def test_cross_exchange_handles_missing_prices() -> None:
    """Fiyat alınamazsa geçersiz olarak işaretlenmeli, çökmemeli."""
    from ..tools.cross_exchange import PriceSources, measure_once

    src = PriceSources(paribu=lambda: {}, binance=lambda s: None)
    m = measure_once(src)
    assert m["gecerli"] is False
    assert m["yol"] == "yok"


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #


def _events(df: pd.DataFrame) -> pd.DataFrame:
    """Test verisi için meta etiket tablosu üretir."""
    tv = get_target_volatility(df["close"], cfg=_CFG.labeling)
    side = primary_side_rule(df)
    ev_idx = cusum_filter(df["close"], tv * _CFG.labeling.cusum_threshold_mult)
    warm = warmup_bars(_CFG.features)
    ev_idx = ev_idx[ev_idx > df.index[min(warm, len(df) - 1)]]
    return get_meta_labels(df, side, event_index=ev_idx, target_vol=tv, cfg=_CFG.labeling)


def _dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test için hizalanmış (X, events, weights) üçlüsü."""
    X = make_features(df, cfg=_CFG.features)
    events = _events(df)
    Xe = X.reindex(events.index)
    usable = Xe.notna().mean(axis=1) > 0.5
    events = events.loc[usable.to_numpy()]
    Xe = Xe.loc[events.index]
    w = get_sample_weights(df["close"], events, cfg=_CFG.labeling)
    return Xe, events, w


def main() -> int:
    """pytest olmadan tüm testleri çalıştırır.

    Returns:
        Başarısız test sayısı (çıkış kodu olarak kullanılır).
    """
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK]   {name}")
        except Exception as exc:  # noqa: BLE001 - test koşucusu tüm hataları raporlar
            failed += 1
            print(f"  [HATA] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} test geçti.")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
