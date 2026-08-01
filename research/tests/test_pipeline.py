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
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import run_backtest, total_cost_rate
from ..config import ResearchConfig
from ..data import (
    fetch_orderbook,
    fetch_paribu_ohlcv,
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
    """Canlı API metotları bilinçli olarak NotImplementedError atmalı."""
    for fn, args in ((fetch_paribu_ohlcv, ("BTC_TL",)), (fetch_orderbook, ("BTC_TL",))):
        try:
            fn(*args)
        except NotImplementedError:
            continue
        raise AssertionError(f"{fn.__name__} NotImplementedError atmalıydı.")


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
