"""Dixon-Coles model matematiği testleri."""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fpredict import model


# --------------------------------------------------------------------------- #
# Yardımcı: bilinen parametrelerden sentetik lig üret
# --------------------------------------------------------------------------- #
def _synthetic_league(seed=1, reps=25):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    attack = dict(zip(teams, [0.6, 0.3, 0.1, -0.1, -0.4, -0.5]))
    defense = dict(zip(teams, [0.5, 0.2, 0.0, -0.1, -0.3, -0.3]))
    home_adv = 0.30
    rows, d = [], date(2023, 1, 1)
    for _ in range(reps):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = math.exp(attack[h] - defense[a] + home_adv)
                mu = math.exp(attack[a] - defense[h])
                rows.append({
                    "date": d, "home_team": h, "away_team": a,
                    "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                })
                d += timedelta(days=1)
    return pd.DataFrame(rows), attack, defense, home_adv


# --------------------------------------------------------------------------- #
# tau düzeltmesi
# --------------------------------------------------------------------------- #
def test_dc_tau_corners():
    lam, mu, rho = 1.4, 1.1, -0.1
    assert np.isclose(model.dc_tau(0, 0, lam, mu, rho), 1 - lam * mu * rho)
    assert np.isclose(model.dc_tau(0, 1, lam, mu, rho), 1 + lam * rho)
    assert np.isclose(model.dc_tau(1, 0, lam, mu, rho), 1 + mu * rho)
    assert np.isclose(model.dc_tau(1, 1, lam, mu, rho), 1 - rho)
    # 2x2 dışındaki tüm hücreler 1
    assert np.isclose(model.dc_tau(2, 3, lam, mu, rho), 1.0)
    assert np.isclose(model.dc_tau(0, 2, lam, mu, rho), 1.0)


# --------------------------------------------------------------------------- #
# zaman ağırlığı
# --------------------------------------------------------------------------- #
def test_time_weights_halflife():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-07-01", "2022-07-01"]))
    ref = date(2023, 1, 1)
    w = model.time_weights(dates, ref, half_life=180.0)
    assert np.isclose(w[0], 1.0)              # referans günü -> ağırlık 1
    # bir yarı-ömür (180 gün) önce -> ~0.5
    assert np.isclose(w[2], 0.5, atol=0.05)
    # gelecekteki maç ağırlığı 1'e sabitlenir (negatif yaş kırpılır)
    assert np.isclose(w[1], 1.0)


def test_time_weights_zero_halflife_is_uniform():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2020-01-01"]))
    w = model.time_weights(dates, date(2023, 1, 1), half_life=0)
    assert np.allclose(w, 1.0)


# --------------------------------------------------------------------------- #
# fit + tahmin
# --------------------------------------------------------------------------- #
def test_fit_recovers_ordering_and_home_adv():
    df, attack, defense, home_adv = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    # ev avantajı makul aralıkta geri kazanılmalı
    assert abs(m.home_adv - home_adv) < 0.12
    # attack sıralaması korunmalı
    est_order = sorted(attack, key=lambda t: -m.attack[t])
    true_order = sorted(attack, key=lambda t: -attack[t])
    assert est_order == true_order


def test_predict_probabilities_sum_to_one():
    df, *_ = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    pred = m.predict("A", "F")
    s = pred["prob_home"] + pred["prob_draw"] + pred["prob_away"]
    assert np.isclose(s, 1.0, atol=1e-6)
    # BTTS ve O/U ikilileri de 1'e toplanmalı
    assert np.isclose(pred["prob_btts_yes"] + pred["prob_btts_no"], 1.0, atol=1e-6)
    assert np.isclose(pred["prob_over25"] + pred["prob_under25"], 1.0, atol=1e-6)


def test_strong_home_team_favored():
    df, *_ = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    pred = m.predict("A", "F")  # en güçlü ev sahibi vs en zayıf deplasman
    assert pred["prob_home"] > pred["prob_away"]
    assert pred["prob_home"] > 0.5
    assert pred["exp_home_goals"] > pred["exp_away_goals"]


def test_home_advantage_effect():
    df, *_ = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    # Aynı eşleşme ev/deplasman yer değiştirince ev sahibi lehine kaymalı
    ab = m.predict("C", "D")
    ba = m.predict("D", "C")
    assert ab["prob_home"] > ba["prob_away"]


def test_squad_boost_reduces_goals():
    df, *_ = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    base = m.predict("A", "F")
    weakened = m.predict("A", "F", home_boost=0.7)
    assert weakened["exp_home_goals"] < base["exp_home_goals"]
    assert weakened["prob_home"] < base["prob_home"]


def test_fit_raises_on_insufficient_data():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"] * 5),
        "home_team": ["A"] * 5, "away_team": ["B"] * 5,
        "home_goals": [1] * 5, "away_goals": [0] * 5,
    })
    with pytest.raises(ValueError):
        model.fit(df, min_matches=30)


def test_predict_unknown_team_raises():
    df, *_ = _synthetic_league()
    m = model.fit(df, half_life=0, min_matches=30)
    with pytest.raises(KeyError):
        m.predict("A", "Nonexistent")
