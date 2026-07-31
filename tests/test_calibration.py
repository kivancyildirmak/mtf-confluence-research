"""Olasılık kalibrasyonu (sıcaklık ölçekleme) testleri."""
import numpy as np
import pandas as pd
import pytest

from fpredict import calibration as cal


# --------------------------------------------------------------------------- #
# Sıcaklık dönüşümü
# --------------------------------------------------------------------------- #
def test_temperature_one_is_identity():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    out = cal.apply_temperature(p, 1.0)
    assert np.allclose(out, p)


def test_temperature_output_sums_to_one():
    p = np.array([[0.7, 0.2, 0.1]])
    for t in (0.5, 1.0, 1.5, 3.0):
        assert np.isclose(cal.apply_temperature(p, t).sum(), 1.0)


def test_high_temperature_reduces_confidence():
    """T>1 güveni azaltmalı ama sıralamayı bozmamalı."""
    p = np.array([0.76, 0.15, 0.09])
    out = cal.apply_temperature(p, 1.5)
    assert out[0] < p[0]                     # tepe olasılık düştü
    assert out[2] > p[2]                     # düşük olasılık yükseldi
    assert np.argmax(out) == np.argmax(p)    # sıralama korundu
    assert list(np.argsort(out)) == list(np.argsort(p))


def test_low_temperature_increases_confidence():
    p = np.array([0.5, 0.3, 0.2])
    out = cal.apply_temperature(p, 0.6)
    assert out[0] > p[0]
    assert np.argmax(out) == np.argmax(p)


def test_temperature_accepts_single_vector_and_matrix():
    single = cal.apply_temperature(np.array([0.6, 0.4]), 2.0)
    assert single.shape == (2,)
    matrix = cal.apply_temperature(np.array([[0.6, 0.4], [0.9, 0.1]]), 2.0)
    assert matrix.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Sıcaklık öğrenme
# --------------------------------------------------------------------------- #
def _overconfident_data(n=800, true_p=0.60, stated_p=0.80, seed=0):
    """Model %80 diyor ama gerçekte %60 tutuyor — aşırı güven senaryosu."""
    rng = np.random.default_rng(seed)
    probs = np.tile([stated_p, (1 - stated_p) / 2, (1 - stated_p) / 2], (n, 1))
    wins = rng.random(n) < true_p
    y = np.zeros((n, 3))
    y[wins, 0] = 1.0
    # kaybedenleri kalan iki sınıfa dağıt
    losers = np.where(~wins)[0]
    y[losers[0::2], 1] = 1.0
    y[losers[1::2], 2] = 1.0
    return probs, y


def test_fit_temperature_detects_overconfidence():
    probs, y = _overconfident_data()
    t = cal.fit_temperature(probs, y)
    assert t > 1.05, f"aşırı güven için T>1 beklenir, bulunan {t}"


def test_fit_temperature_improves_logloss():
    probs, y = _overconfident_data()
    t = cal.fit_temperature(probs, y)
    before = cal._nll(probs, y)
    after = cal._nll(cal.apply_temperature(probs, t), y)
    assert after < before


def test_fit_temperature_leaves_calibrated_data_alone():
    """Zaten kalibre veride sıcaklık ~1 kalmalı."""
    probs, y = _overconfident_data(true_p=0.80, stated_p=0.80, seed=3)
    t = cal.fit_temperature(probs, y)
    assert 0.85 < t < 1.2


def test_fit_temperature_ignores_tiny_samples():
    probs = np.array([[0.8, 0.1, 0.1]] * 5)
    y = np.array([[1.0, 0, 0]] * 5)
    assert cal.fit_temperature(probs, y) == 1.0


# --------------------------------------------------------------------------- #
# Backtest kayıtlarından öğrenme
# --------------------------------------------------------------------------- #
def _records(n=400, seed=1):
    rng = np.random.default_rng(seed)
    p_home = rng.uniform(0.55, 0.85, n)
    rest = 1 - p_home
    rows = []
    for ph, r in zip(p_home, rest):
        # gerçek olasılık, iddia edilenden düşük (aşırı güven)
        real = ph * 0.85
        u = rng.random()
        actual = "H" if u < real else ("D" if u < real + (1 - real) / 2 else "A")
        hg, ag = (2, 0) if actual == "H" else ((1, 1) if actual == "D" else (0, 2))
        rows.append({
            "date": pd.Timestamp("2024-01-01"), "home": "A", "away": "B",
            "actual": actual, "p_home": ph, "p_draw": r / 2, "p_away": r / 2,
            "pred": "H",
            "p_over25": 0.75, "over25": int(hg + ag >= 3),
            "p_btts": 0.70, "btts": int(hg >= 1 and ag >= 1),
        })
    return pd.DataFrame(rows)


def test_fit_from_backtest_returns_calibration():
    c = cal.fit_from_backtest(_records(), "NOR")
    assert c.league == "NOR"
    assert c.n_matches == 400
    assert c.t_1x2 > 1.0                       # aşırı güven -> yumuşatma
    assert c.logloss_after <= c.logloss_before
    assert c.fitted_at
    assert "T=" in c.describe()


def test_fit_from_backtest_calibrates_other_markets():
    c = cal.fit_from_backtest(_records(), "NOR")
    # Üst/Alt ve KG için de sıcaklık öğrenilmeli (varsayılan 1.0'dan farklı)
    assert c.t_over25 != 1.0
    assert c.t_btts != 1.0


def test_fit_from_backtest_rejects_small_sample():
    with pytest.raises(ValueError, match="yetersiz"):
        cal.fit_from_backtest(_records(n=20), "NOR")


def test_fit_from_backtest_rejects_missing_columns():
    df = _records().drop(columns=["p_draw"])
    with pytest.raises(ValueError, match="eksik"):
        cal.fit_from_backtest(df, "NOR")


# --------------------------------------------------------------------------- #
# Tahmine uygulama
# --------------------------------------------------------------------------- #
def _pred():
    return {
        "home": "Bodo/Glimt", "away": "Lillestrom",
        "prob_home": 0.762, "prob_draw": 0.150, "prob_away": 0.087,
        "prob_over25": 0.617, "prob_under25": 0.383,
        "prob_btts_yes": 0.464, "prob_btts_no": 0.536,
        "exp_home_goals": 2.48, "exp_away_goals": 0.71,
        "most_likely_score": (2, 0),
    }


def test_calibrate_prediction_softens_and_normalizes():
    c = cal.Calibration(league="NOR", t_1x2=1.4, t_over25=1.2, t_btts=1.1)
    out = cal.calibrate_prediction(_pred(), c)

    assert out["prob_home"] < 0.762                    # aşırı güven geri çekildi
    assert np.isclose(out["prob_home"] + out["prob_draw"] + out["prob_away"], 1.0)
    assert np.isclose(out["prob_over25"] + out["prob_under25"], 1.0)
    assert np.isclose(out["prob_btts_yes"] + out["prob_btts_no"], 1.0)
    assert out["calibrated"] is True


def test_calibrate_prediction_preserves_goals_and_score():
    """Beklenen goller ve en olası skor kalibrasyondan etkilenmemeli."""
    c = cal.Calibration(league="NOR", t_1x2=1.6)
    out = cal.calibrate_prediction(_pred(), c)
    assert out["exp_home_goals"] == 2.48
    assert out["exp_away_goals"] == 0.71
    assert out["most_likely_score"] == (2, 0)


def test_calibrate_prediction_preserves_favourite():
    c = cal.Calibration(league="NOR", t_1x2=2.5)
    out = cal.calibrate_prediction(_pred(), c)
    assert out["prob_home"] > out["prob_draw"] > out["prob_away"]


def test_calibrate_prediction_without_calibration_is_noop():
    p = _pred()
    assert cal.calibrate_prediction(p, None) == p


def test_identity_calibration_flagged():
    assert cal.Calibration(league="X").is_identity
    assert not cal.Calibration(league="X", t_1x2=1.4).is_identity


# --------------------------------------------------------------------------- #
# Güvenilirlik tablosu
# --------------------------------------------------------------------------- #
def test_reliability_table_shape():
    rows = cal.reliability_table(_records())
    assert rows
    for r in rows:
        assert {"bant", "maç", "ortalama_tahmin", "gerçekleşme"} <= set(r)
        assert 0 <= r["gerçekleşme"] <= 100


def test_reliability_table_after_calibration_closes_gap():
    """Kalibrasyon sonrası tahmin ile gerçekleşme arasındaki fark azalmalı."""
    rec = _records(n=1200, seed=5)
    c = cal.fit_from_backtest(rec, "NOR")

    def mean_gap(rows):
        return np.mean([abs(r["ortalama_tahmin"] - r["gerçekleşme"]) for r in rows])

    before = mean_gap(cal.reliability_table(rec, 1.0))
    after = mean_gap(cal.reliability_table(rec, c.t_1x2))
    assert after < before


# --------------------------------------------------------------------------- #
# Kayıt / yükleme
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path, monkeypatch):
    from fpredict import config
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    assert cal.load("NOR") is None
    c = cal.Calibration(league="NOR", t_1x2=1.33, n_matches=2676)
    cal.save(c)

    loaded = cal.load("NOR")
    assert loaded is not None
    assert loaded.t_1x2 == 1.33
    assert loaded.n_matches == 2676
    assert cal.load("SWE") is None          # diğer ligler etkilenmez


def test_save_does_not_clobber_other_settings(tmp_path, monkeypatch):
    from fpredict import config, squad_adjust
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    squad_adjust.save_api_key("SECRET")
    cal.save(cal.Calibration(league="NOR", t_1x2=1.2))
    assert squad_adjust.load_api_key() == "SECRET"
    assert cal.load("NOR").t_1x2 == 1.2


def test_clear_removes_only_target_league(tmp_path, monkeypatch):
    from fpredict import config
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    cal.save(cal.Calibration(league="NOR", t_1x2=1.2))
    cal.save(cal.Calibration(league="SWE", t_1x2=1.1))
    cal.clear("NOR")
    assert cal.load("NOR") is None
    assert cal.load("SWE") is not None


# --------------------------------------------------------------------------- #
# Sabit üyelikli karşılaştırma (bantlar arası kayma sorunu)
# --------------------------------------------------------------------------- #
def test_comparison_bands_have_fixed_membership():
    """Bantlar HAM olasılığa göre kurulmalı; kalibrasyon maç sayısını değiştirmemeli."""
    rec = _records(n=600, seed=7)
    c = cal.fit_from_backtest(rec, "NOR")

    rows_id = cal.calibration_comparison(rec, 1.0)      # kimlik dönüşümü
    rows_cal = cal.calibration_comparison(rec, c.t_1x2)

    assert [r["bant"] for r in rows_id] == [r["bant"] for r in rows_cal]
    assert [r["maç"] for r in rows_id] == [r["maç"] for r in rows_cal]
    # Gerçekleşme oranı da değişmemeli (sıralama korunduğu için isabet aynıdır)
    assert [r["gerçekleşme"] for r in rows_id] == [r["gerçekleşme"] for r in rows_cal]


def test_comparison_shows_calibration_moves_toward_reality():
    """Aşırı güvenli veride kalibre tahmin gerçekleşmeye yaklaşmalı."""
    rec = _records(n=1200, seed=11)
    c = cal.fit_from_backtest(rec, "NOR")
    rows = cal.calibration_comparison(rec, c.t_1x2)

    improved = [r for r in rows if r["kalibre_fark"] <= r["ham_fark"]]
    assert len(improved) >= len(rows) - 1        # neredeyse tüm bantlarda iyileşme
    # Toplamda ortalama sapma azalmalı
    ham = np.mean([r["ham_fark"] for r in rows])
    kal = np.mean([r["kalibre_fark"] for r in rows])
    assert kal < ham


def test_comparison_identity_temperature_is_noop():
    rec = _records(n=300, seed=2)
    rows = cal.calibration_comparison(rec, 1.0)
    for r in rows:
        assert r["ham_tahmin"] == r["kalibre_tahmin"]
        assert r["ham_fark"] == r["kalibre_fark"]


def test_comparison_reports_match_counts():
    rec = _records(n=500, seed=4)
    rows = cal.calibration_comparison(rec, 1.3)
    assert sum(r["maç"] for r in rows) <= len(rec)
    assert all(r["maç"] >= 10 for r in rows)     # küçük bantlar elenir
