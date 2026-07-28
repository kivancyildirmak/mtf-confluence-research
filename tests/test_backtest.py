"""Backtest ve değer modülü testleri."""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fpredict import backtest, value


def _synthetic_matches(reps=20, seed=3):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    attack = dict(zip(teams, [0.5, 0.3, 0.0, -0.1, -0.3, -0.4]))
    defense = dict(zip(teams, [0.4, 0.2, 0.0, -0.1, -0.2, -0.3]))
    rows, d = [], date(2022, 1, 1)
    for _ in range(reps):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = math.exp(attack[h] - defense[a] + 0.28)
                mu = math.exp(attack[a] - defense[h])
                hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
                rows.append({
                    "date": d, "home_team": h, "away_team": a,
                    "home_goals": hg, "away_goals": ag,
                    "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                })
                d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_backtest_beats_uniform_baseline():
    df = _synthetic_matches()
    res = backtest.run_backtest(df, half_life=0, min_train=200, refit_every=30)
    assert res.n_predicted > 0
    # Kalibre model bilgisiz 1/3 tahminden daha düşük log-loss vermeli
    assert res.log_loss < res.baseline_uniform_logloss
    # Metrikler geçerli aralıkta
    assert 0.0 <= res.accuracy <= 1.0
    assert res.brier >= 0.0
    assert isinstance(res.commentary(), str) and res.commentary()


def test_backtest_no_leakage_ordering():
    # Backtest kronolojik sırada çalışmalı; kayıtlar tarih artışıyla gelmeli
    df = _synthetic_matches()
    res = backtest.run_backtest(df, half_life=0, min_train=200, refit_every=50)
    dates = res.records["date"].tolist()
    assert dates == sorted(dates)


# --------------------------------------------------------------------------- #
# value modülü
# --------------------------------------------------------------------------- #
def test_implied_probabilities_normalized():
    imp = value.implied_probabilities(2.0, 3.5, 4.0)
    assert imp is not None
    assert np.isclose(sum(imp), 1.0)


def test_implied_probabilities_missing_returns_none():
    assert value.implied_probabilities(None, 3.5, 4.0) is None


def test_expected_value_sign():
    # Model %60 diyor, oran 2.0 (ima %50) => pozitif EV
    assert value.expected_value(0.60, 2.0) > 0
    # Model %30 diyor, oran 2.0 => negatif EV
    assert value.expected_value(0.30, 2.0) < 0


def test_find_value_bets_flags_edge():
    preds = [{
        "home": "A", "away": "B",
        "prob_home": 0.60, "prob_draw": 0.25, "prob_away": 0.15,
        "odds_h": 2.20, "odds_d": 3.4, "odds_a": 5.0,  # ev ima ~0.45 < model 0.60
    }]
    vb = value.find_value_bets(preds, min_edge=0.05)
    assert not vb.empty
    assert "1 (Ev)" in vb["bahis"].values
    assert (vb["EV_%"] > 0).all()
