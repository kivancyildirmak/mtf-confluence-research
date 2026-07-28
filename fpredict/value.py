"""Değer (value) bahis karşılaştırması.

Modelin ürettiği olasılıklar, CSV'deki bahis oranlarının ima ettiği olasılıklarla
kıyaslanır. Bir bahsin "değerli" sayılması için modelin olasılığı, bahiscinin
(marj arındırılmış) ima ettiği olasılıktan belirgin biçimde yüksek olmalıdır.

Ana kavramlar
-------------
* Bahisçi oranı O için ham ima olasılık = 1/O.
* 1X2'nin üç ham olasılığı toplamı 1'i aşar (overround/marj). Bunu normalize
  ederek (her birini toplama bölerek) marjı kabaca arındırırız.
* Beklenen değer (EV) = p_model * (O - 1) - (1 - p_model).
  EV > 0 ise model o bahsi "değerli" görür.

UYARI: Pozitif EV, kazanç garantisi DEĞİLDİR. Model yanlış olabilir, oranlar
piyasa tarafından verimli fiyatlanmış olabilir. Bu ekran yalnızca eğitim/analiz
amaçlıdır.
"""
from __future__ import annotations

import pandas as pd


def implied_probabilities(odds_h, odds_d, odds_a):
    """1X2 oranlarından marj-arındırılmış ima olasılıklarını döndürür.

    Eksik oran varsa (None) None döner.
    """
    if not odds_h or not odds_d or not odds_a:
        return None
    raw = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    total = sum(raw)
    if total <= 0:
        return None
    return [r / total for r in raw]  # normalize => marj arındırma (basit yöntem)


def expected_value(p_model: float, odds: float) -> float:
    """1 birim bahis için beklenen değer."""
    return p_model * (odds - 1.0) - (1.0 - p_model)


def find_value_bets(
    predictions: list[dict],
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """Tahmin listesinden değerli 1X2 bahislerini süzer.

    Args:
        predictions: her biri şu anahtarları içeren dict listesi:
            home, away, prob_home, prob_draw, prob_away,
            odds_h, odds_d, odds_a
        min_edge: model olasılığı - ima olasılık farkı için minimum eşik.

    Returns:
        Değerli bahisleri EV'ye göre azalan sıralı DataFrame.
    """
    rows = []
    for pr in predictions:
        imp = implied_probabilities(pr.get("odds_h"), pr.get("odds_d"), pr.get("odds_a"))
        if imp is None:
            continue
        legs = [
            ("1 (Ev)", pr["prob_home"], imp[0], pr.get("odds_h")),
            ("X (Beraberlik)", pr["prob_draw"], imp[1], pr.get("odds_d")),
            ("2 (Deplasman)", pr["prob_away"], imp[2], pr.get("odds_a")),
        ]
        for label, p_model, p_imp, odds in legs:
            edge = p_model - p_imp
            ev = expected_value(p_model, odds)
            if edge >= min_edge and ev > 0:
                rows.append({
                    "maç": f"{pr['home']} - {pr['away']}",
                    "bahis": label,
                    "model_olasılık": round(p_model, 3),
                    "ima_olasılık": round(p_imp, 3),
                    "edge": round(edge, 3),
                    "oran": odds,
                    "EV_%": round(ev * 100, 1),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("EV_%", ascending=False).reset_index(drop=True)
    return df
