"""Olasılık kalibrasyonu — sıcaklık ölçekleme (temperature scaling).

Sorun
-----
Dixon-Coles modeli düşük/orta olasılıklarda iyi kalibredir, ancak yüksek
olasılıklarda **fazla iddialıdır**. Norveç Eliteserien üzerinde 2.676 maçlık
walk-forward backtest'te ölçülen davranış:

    model %0-40 dedi  -> gerçekte %38 oldu   (iyi)
    model %40-50 dedi -> gerçekte %43 oldu   (iyi)
    model %50-60 dedi -> gerçekte %53 oldu   (iyi)
    model %60-70 dedi -> gerçekte %60 oldu   (iyi)
    model %70+  dedi  -> gerçekte %68 oldu   (FAZLA İDDİALI)

Kullanıcı bir bahis kararı verirken gösterilen yüzdenin gerçekleşme oranına
yakın olması gerekir; aksi halde model kendi belirsizliğini gizler.

Yöntem
------
Sıcaklık ölçekleme, olasılıkları tek bir T parametresiyle yumuşatır:

    p_kalibre ∝ p ** (1 / T)      (sonra toplamı 1'e normalize edilir)

    T = 1  -> değişiklik yok
    T > 1  -> olasılıklar 1/k'ya doğru çekilir (güven azalır)
    T < 1  -> keskinleşir (güven artar)

T, backtest kayıtlarından log-olabilirliği en büyükleyecek şekilde bulunur.
Tek parametreli olduğu için aşırı uyum (overfitting) riski çok düşüktür ve
tahminlerin SIRALAMASINI hiç değiştirmez — yalnızca güven düzeyini düzeltir.

Neden backtest kayıtları güvenli
--------------------------------
Backtest her maçı yalnızca ondan ÖNCEKİ verilerle tahmin eder. Dolayısıyla bu
kayıtlar örneklem-dışıdır ve kalibrasyon için uygun bir eğitim kümesidir.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize_scalar

from . import config

_EPS = 1e-12
_T_BOUNDS = (0.3, 5.0)   # makul sıcaklık aralığı


# --------------------------------------------------------------------------- #
# Çekirdek dönüşüm
# --------------------------------------------------------------------------- #
def apply_temperature(probs, temperature: float):
    """Olasılıklara sıcaklık ölçekleme uygular ve normalize eder.

    Args:
        probs: (n, k) dizi ya da tek bir olasılık vektörü.
        temperature: T > 0. 1.0 kimlik dönüşümüdür.

    Returns:
        Aynı biçimde, satır toplamları 1 olan kalibre olasılıklar.
    """
    p = np.asarray(probs, dtype=float)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    p = np.clip(p, _EPS, 1.0)
    if temperature <= 0:
        temperature = 1.0
    q = p ** (1.0 / temperature)
    q = q / q.sum(axis=1, keepdims=True)
    return q[0] if single else q


def _nll(probs, y_onehot) -> float:
    """Ortalama negatif log-olabilirlik (log-loss)."""
    p = np.clip(probs, _EPS, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


def fit_temperature(probs, y_onehot) -> float:
    """Log-loss'u en küçükleyen sıcaklığı bulur (tek parametreli arama)."""
    probs = np.asarray(probs, dtype=float)
    y_onehot = np.asarray(y_onehot, dtype=float)
    if len(probs) < 30:
        return 1.0  # yetersiz veri: dokunma

    def objective(t):
        return _nll(apply_temperature(probs, t), y_onehot)

    res = minimize_scalar(objective, bounds=_T_BOUNDS, method="bounded")
    if not res.success:
        return 1.0
    return float(res.x)


def _binary_matrix(p_yes):
    """Tek olasılıktan iki sütunlu matris ([evet, hayır])."""
    p = np.clip(np.asarray(p_yes, dtype=float), _EPS, 1 - _EPS)
    return np.column_stack([p, 1.0 - p])


def _binary_onehot(y):
    y = np.asarray(y, dtype=int)
    return np.column_stack([y, 1 - y])


# --------------------------------------------------------------------------- #
# Kalibrasyon kaydı
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """Bir lig için öğrenilmiş sıcaklıklar ve öncesi/sonrası ölçümler."""
    league: str
    t_1x2: float = 1.0
    t_over25: float = 1.0
    t_btts: float = 1.0
    n_matches: int = 0
    fitted_at: str = ""
    logloss_before: float = float("nan")
    logloss_after: float = float("nan")

    @property
    def is_identity(self) -> bool:
        """Kalibrasyon pratikte etkisiz mi?"""
        return all(abs(t - 1.0) < 0.02
                   for t in (self.t_1x2, self.t_over25, self.t_btts))

    def describe(self) -> str:
        yon = "güveni azaltıyor" if self.t_1x2 > 1 else "güveni artırıyor"
        return (
            f"1X2 sıcaklığı T={self.t_1x2:.2f} ({yon}), "
            f"Üst/Alt T={self.t_over25:.2f}, KG T={self.t_btts:.2f} — "
            f"{self.n_matches} maçtan öğrenildi."
        )


def fit_from_backtest(records, league: str) -> Calibration:
    """Backtest kayıtlarından bir lig için kalibrasyon öğrenir.

    Args:
        records: backtest.run_backtest(...).records DataFrame'i.
        league: lig kodu (kaydetme anahtarı).

    Raises:
        ValueError: kayıtlar eksik veya yetersizse.
    """
    need = {"p_home", "p_draw", "p_away", "actual"}
    missing = need - set(records.columns)
    if missing:
        raise ValueError(f"Backtest kayıtlarında eksik sütun: {missing}")
    if len(records) < 50:
        raise ValueError(
            f"Kalibrasyon için yetersiz kayıt: {len(records)} (en az 50 gerekir). "
            "Backtest'i daha uzun bir dönemde çalıştırın."
        )

    # --- 1X2 --------------------------------------------------------------- #
    probs = records[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcomes = records["actual"].to_numpy()
    y = np.column_stack([outcomes == "H", outcomes == "D", outcomes == "A"]).astype(float)

    t_1x2 = fit_temperature(probs, y)
    ll_before = _nll(probs, y)
    ll_after = _nll(apply_temperature(probs, t_1x2), y)

    # --- Üst/Alt 2.5 ve KG (varsa) ------------------------------------------ #
    t_ou, t_btts = 1.0, 1.0
    if "p_over25" in records.columns and "over25" in records.columns:
        sub = records.dropna(subset=["p_over25", "over25"])
        if len(sub) >= 50:
            t_ou = fit_temperature(
                _binary_matrix(sub["p_over25"]), _binary_onehot(sub["over25"])
            )
    if "p_btts" in records.columns and "btts" in records.columns:
        sub = records.dropna(subset=["p_btts", "btts"])
        if len(sub) >= 50:
            t_btts = fit_temperature(
                _binary_matrix(sub["p_btts"]), _binary_onehot(sub["btts"])
            )

    return Calibration(
        league=league,
        t_1x2=t_1x2,
        t_over25=t_ou,
        t_btts=t_btts,
        n_matches=len(records),
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        logloss_before=ll_before,
        logloss_after=ll_after,
    )


# --------------------------------------------------------------------------- #
# Tahmine uygulama
# --------------------------------------------------------------------------- #
def calibrate_prediction(pred: dict, cal: Calibration | None) -> dict:
    """Bir tahmin sözlüğünü kalibre eder (kopya döndürür).

    Yalnızca OLASILIKLAR düzeltilir. Beklenen gol sayıları ve en olası skor
    modelin ham çıktısıdır; sıcaklık ölçekleme bunlara uygulanmaz (anlamlı bir
    karşılığı yoktur). Arayüz bunu kullanıcıya belirtir.
    """
    if cal is None:
        return pred
    out = dict(pred)

    p = apply_temperature(
        np.array([pred["prob_home"], pred["prob_draw"], pred["prob_away"]]),
        cal.t_1x2,
    )
    out["prob_home"], out["prob_draw"], out["prob_away"] = (float(x) for x in p)

    ou = apply_temperature(np.array([pred["prob_over25"], pred["prob_under25"]]),
                           cal.t_over25)
    out["prob_over25"], out["prob_under25"] = float(ou[0]), float(ou[1])

    bt = apply_temperature(np.array([pred["prob_btts_yes"], pred["prob_btts_no"]]),
                           cal.t_btts)
    out["prob_btts_yes"], out["prob_btts_no"] = float(bt[0]), float(bt[1])

    out["calibrated"] = True
    return out


# --------------------------------------------------------------------------- #
# Güvenilirlik tablosu (arayüzde öncesi/sonrası göstermek için)
# --------------------------------------------------------------------------- #
def calibration_comparison(records, temperature: float):
    """Kalibrasyonun etkisini AYNI maçlar üzerinde karşılaştırır.

    `reliability_table` bantları kendi olasılığına göre kurar; kalibrasyon
    olasılıkları sıkıştırdığı için maçlar bantlar arasında yer değiştirir ve
    "önce/sonra" sütunları farklı maç kümelerini gösterir — yan yana konursa
    yanıltıcı olur (hatta kalibrasyon iyileştirmişken kötüleşmiş gibi görünür).

    Bu fonksiyon bantları HAM olasılığa göre sabitler. Sıcaklık ölçekleme
    sıralamayı değiştirmediği için isabet oranı iki durumda da aynıdır; böylece
    tek soru kalır: gösterilen yüzde gerçekleşmeye yaklaştı mı?

    Returns:
        [{"bant", "maç", "ham_tahmin", "kalibre_tahmin", "gerçekleşme",
          "ham_fark", "kalibre_fark"}, ...]
    """
    probs = records[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    cal_probs = apply_temperature(probs, temperature)
    outcomes = records["actual"].to_numpy()
    labels = np.array(["H", "D", "A"])

    idx = probs.argmax(axis=1)          # sıralama değişmez -> tek indeks yeter
    p_raw = probs.max(axis=1)
    p_cal = cal_probs[np.arange(len(cal_probs)), idx]
    hit = labels[idx] == outcomes

    rows = []
    for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
        m = (p_raw >= lo) & (p_raw < hi)
        if m.sum() < 10:
            continue
        ham = float(p_raw[m].mean()) * 100
        kal = float(p_cal[m].mean()) * 100
        ger = float(hit[m].mean()) * 100
        rows.append({
            "bant": f"%{int(lo*100)}–{int(min(hi, 1.0)*100)}",
            "maç": int(m.sum()),
            "ham_tahmin": round(ham, 1),
            "kalibre_tahmin": round(kal, 1),
            "gerçekleşme": round(ger, 1),
            "ham_fark": round(abs(ham - ger), 1),
            "kalibre_fark": round(abs(kal - ger), 1),
        })
    return rows


def reliability_table(records, temperature: float = 1.0):
    """Model güveni ile gerçekleşme oranını bantlar halinde karşılaştırır.

    Returns:
        [{"bant", "maç", "ortalama_tahmin", "gerçekleşme"}, ...]
    """
    probs = records[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    if temperature != 1.0:
        probs = apply_temperature(probs, temperature)
    outcomes = records["actual"].to_numpy()
    labels = np.array(["H", "D", "A"])

    idx = probs.argmax(axis=1)
    p_max = probs.max(axis=1)
    hit = labels[idx] == outcomes

    rows = []
    for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
        m = (p_max >= lo) & (p_max < hi)
        if m.sum() < 10:
            continue
        rows.append({
            "bant": f"%{int(lo*100)}–{int(min(hi, 1.0)*100)}",
            "maç": int(m.sum()),
            "ortalama_tahmin": round(float(p_max[m].mean()) * 100, 1),
            "gerçekleşme": round(float(hit[m].mean()) * 100, 1),
        })
    return rows


# --------------------------------------------------------------------------- #
# Diske kayıt (settings.json)
# --------------------------------------------------------------------------- #
def save(cal: Calibration) -> None:
    config.ensure_app_dir()
    settings = _load_settings()
    settings.setdefault("calibration", {})[cal.league] = asdict(cal)
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def load(league: str) -> Calibration | None:
    data = _load_settings().get("calibration", {}).get(league)
    if not data:
        return None
    try:
        return Calibration(**data)
    except TypeError:
        return None


def clear(league: str) -> None:
    settings = _load_settings()
    settings.get("calibration", {}).pop(league, None)
    config.ensure_app_dir()
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def _load_settings() -> dict:
    if config.SETTINGS_PATH.exists():
        try:
            return json.loads(config.SETTINGS_PATH.read_text())
        except (ValueError, OSError):
            return {}
    return {}
