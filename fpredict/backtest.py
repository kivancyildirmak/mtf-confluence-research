"""Walk-forward (ileriye doğru) geçmişe dönük test.

Her maç, YALNIZCA kendisinden önce oynanmış maçlarla fit edilen bir modelle
tahmin edilir (data leakage yok). Metrikler:

    * accuracy  — en yüksek olasılıklı sonucun gerçekleşme oranı
    * log_loss  — çok-sınıflı log kayıp (düşük = iyi)
    * brier     — çok-sınıflı Brier skoru (düşük = iyi)

Kıyaslama için iki naif temel de hesaplanır:
    * "her zaman ev sahibi" (favori proxy'si)
    * "eşit 1/3-1/3-1/3" (rastgele/bilgisiz)

Performans notu: her maç için yeniden fit etmek pahalıdır. Bu yüzden model
her `refit_every` maçta bir yeniden fit edilir; arada aynı katsayılar kullanılır
(pratik ve yaygın bir yaklaşım). En eski `min_train` maç eğitim ısınması olarak
atlanır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, model


_OUTCOMES = ["H", "D", "A"]


@dataclass
class BacktestResult:
    n_predicted: int
    accuracy: float
    log_loss: float
    brier: float
    baseline_home_acc: float
    baseline_home_logloss: float
    baseline_uniform_logloss: float
    records: pd.DataFrame = field(default_factory=pd.DataFrame)

    def commentary(self) -> str:
        """Sonuçları sade bir dille yorumlar."""
        lines = []
        if self.accuracy > self.baseline_home_acc:
            lines.append(
                f"Model doğruluğu (%{self.accuracy*100:.1f}) 'her zaman ev sahibi' "
                f"temelinden (%{self.baseline_home_acc*100:.1f}) YÜKSEK."
            )
        else:
            lines.append(
                f"Model doğruluğu (%{self.accuracy*100:.1f}) 'her zaman ev sahibi' "
                f"temelini (%{self.baseline_home_acc*100:.1f}) geçemedi — dikkat."
            )
        if self.log_loss < self.baseline_uniform_logloss:
            lines.append(
                f"Log-loss ({self.log_loss:.3f}) bilgisiz 1/3 tahmininden "
                f"({self.baseline_uniform_logloss:.3f}) daha iyi (kalibrasyon anlamlı)."
            )
        else:
            lines.append(
                f"Log-loss ({self.log_loss:.3f}) bilgisiz tahmini "
                f"({self.baseline_uniform_logloss:.3f}) geçemedi."
            )
        if self.log_loss < self.baseline_home_logloss:
            lines.append("Olasılık kalibrasyonu naif ev-sahibi temelinden de iyi.")
        return " ".join(lines)


def _onehot(result: str) -> np.ndarray:
    return np.array([1.0 if result == o else 0.0 for o in _OUTCOMES])


def run_backtest(
    matches: pd.DataFrame,
    half_life: float = config.DEFAULT_HALF_LIFE_DAYS,
    min_train: int = 200,
    refit_every: int = 20,
    max_goals: int = 6,
    progress=None,
) -> BacktestResult:
    """Bir ligin maçları üzerinde walk-forward backtest çalıştırır.

    Args:
        matches: [date, home_team, away_team, home_goals, away_goals, result].
        min_train: tahmine başlamadan önce gereken minimum geçmiş maç.
        refit_every: kaç maçta bir modelin yeniden fit edileceği.
        progress: opsiyonel callback(frac: float).
    """
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "result" not in df.columns or df["result"].isna().any():
        df["result"] = np.where(
            df["home_goals"] > df["away_goals"], "H",
            np.where(df["home_goals"] < df["away_goals"], "A", "D"),
        )

    if len(df) <= min_train + 10:
        raise ValueError(
            f"Backtest için yetersiz maç: {len(df)}. "
            f"En az {min_train + 10} gerekir (min_train={min_train})."
        )

    records = []
    fitted = None
    logloss_sum = brier_sum = correct = 0.0
    base_home_correct = base_home_logloss = base_uniform_logloss = 0.0
    n = 0
    eps = 1e-15

    total = len(df) - min_train
    for i in range(min_train, len(df)):
        if (i - min_train) % refit_every == 0 or fitted is None:
            train = df.iloc[:i]
            try:
                fitted = model.fit(
                    train, ref_date=df.iloc[i]["date"].date(),
                    half_life=half_life, max_goals=max_goals, min_matches=30,
                )
            except ValueError:
                continue
        if progress and total > 0:
            progress((i - min_train) / total)

        row = df.iloc[i]
        home, away, actual = row["home_team"], row["away_team"], row["result"]
        # Eğitimde görülmemiş takım — tahmin edilemez, atla
        if home not in fitted.attack or away not in fitted.attack:
            continue

        pred = fitted.predict(home, away)
        p = np.array([pred["prob_home"], pred["prob_draw"], pred["prob_away"]])
        p = np.clip(p, eps, 1.0)
        p /= p.sum()

        y = _onehot(actual)
        pred_outcome = _OUTCOMES[int(np.argmax(p))]

        correct += (pred_outcome == actual)
        logloss_sum += -np.sum(y * np.log(p))
        brier_sum += np.sum((p - y) ** 2)

        # Temeller
        base_home_correct += (actual == "H")
        base_home_p = np.clip(np.array([0.46, 0.27, 0.27]), eps, 1.0)  # tipik lig dağılımı
        base_home_logloss += -np.sum(y * np.log(base_home_p))
        base_uniform_logloss += -np.sum(y * np.log(np.array([1/3, 1/3, 1/3])))

        n += 1
        total_goals = row["home_goals"] + row["away_goals"]
        records.append({
            "date": row["date"], "home": home, "away": away, "actual": actual,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "pred": pred_outcome,
            # Kalibrasyon için ek pazarlar: tahmin edilen olasılık + gerçekleşme
            "p_over25": pred["prob_over25"],
            "over25": int(total_goals >= 3),
            "p_btts": pred["prob_btts_yes"],
            "btts": int(row["home_goals"] >= 1 and row["away_goals"] >= 1),
        })

    if n == 0:
        raise ValueError("Hiç maç tahmin edilemedi (takım eşleşmesi/veri sorunu).")

    return BacktestResult(
        n_predicted=n,
        accuracy=correct / n,
        log_loss=logloss_sum / n,
        brier=brier_sum / n,
        baseline_home_acc=base_home_correct / n,
        baseline_home_logloss=base_home_logloss / n,
        baseline_uniform_logloss=base_uniform_logloss / n,
        records=pd.DataFrame(records),
    )
