"""Dixon-Coles genişletmeli Poisson gol modeli.

Model
-----
Ev sahibi i, deplasman j maçında beklenen goller:

    log(lambda_ev)  = attack[i] - defense[j] + home_adv
    log(mu_deplas)  = attack[j] - defense[i]

Goller Poisson dağılır; düşük skorlar için Dixon-Coles tau(x, y) düzeltmesi
uygulanır (0-0, 1-0, 0-1, 1-1 hücreleri, bağımsızlık varsayımının en zayıf
olduğu yer):

    tau(0,0) = 1 - lambda*mu*rho
    tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    aksi     = 1

Zaman ağırlığı: her maç, referans tarihe uzaklığına göre üstel azalan bir
ağırlık alır; yarı-ömür (varsayılan 180 gün) ayarlanabilir.

    w(t) = 2 ** (-yaş_gün / yarı_ömür)

Parametreler maksimum (ağırlıklı) olabilirlikle scipy.optimize.minimize ile
bulunur. attack/defense konum belirsizliğini kırmak için attack ortalaması
0'a sabitlenir (fit sonrası yeniden merkezleme).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config


# --------------------------------------------------------------------------- #
# Dixon-Coles düşük-skor düzeltmesi
# --------------------------------------------------------------------------- #
def dc_tau(x, y, lam, mu, rho):
    """Vektörize edilebilir tau düzeltmesi (skaler veya numpy dizisi)."""
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


# --------------------------------------------------------------------------- #
# Zaman ağırlığı
# --------------------------------------------------------------------------- #
def time_weights(match_dates: pd.Series, ref_date: date, half_life: float) -> np.ndarray:
    """Referans tarihe göre üstel azalan ağırlıklar (yarı-ömür gün cinsinden)."""
    ref = pd.Timestamp(ref_date)
    ages = (ref - pd.to_datetime(match_dates)).dt.days.to_numpy(dtype=float)
    ages = np.clip(ages, 0, None)  # gelecekteki maçlar (varsa) ağırlık 1
    if half_life <= 0:
        return np.ones_like(ages)
    return np.power(2.0, -ages / half_life)


# --------------------------------------------------------------------------- #
# Fit edilmiş model
# --------------------------------------------------------------------------- #
@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    max_goals: int = config.DEFAULT_MAX_GOALS
    log_likelihood: float = field(default=float("nan"))
    n_matches: int = 0

    # ----- isim çözümleme -------------------------------------------------- #
    def resolve_team(self, name: str) -> str:
        """Serbest yazılmış bir takım adını modeldeki adla eşleştirir.

        Kaynaklar arası yazım farkları için ('Hamkam' -> 'Ham-Kam',
        'Fenerbahçe' -> 'Fenerbahce') isim eşleştirme katmanını kullanır.

        Raises:
            KeyError: makul bir eşleşme bulunamazsa (mevcut adlardan örnekle).
        """
        if name in self.attack:
            return name
        from .name_matching import best_match

        match, score = best_match(name, self.teams, threshold=0.75)
        if match:
            return match
        raise KeyError(
            f"Takım modelde bulunamadı: {name!r} (en yakın skor {score:.2f}). "
            f"Mevcut takımlardan bazıları: {', '.join(self.teams[:5])}…"
        )

    # ----- gol beklentileri ------------------------------------------------ #
    def expected_goals(self, home: str, away: str, home_boost: float = 1.0,
                       away_boost: float = 1.0) -> tuple[float, float]:
        """Ev ve deplasman için beklenen gol (lambda, mu).

        `home_boost`/`away_boost` kadro/sakatlık güç çarpanlarıdır (1.0 = etkisiz).
        Çarpan doğrudan beklenen gol üzerine uygulanır (kaba yaklaşım).
        """
        home = self.resolve_team(home)
        away = self.resolve_team(away)
        lam = math.exp(self.attack[home] - self.defense[away] + self.home_adv)
        mu = math.exp(self.attack[away] - self.defense[home])
        return lam * home_boost, mu * away_boost

    # ----- skor matrisi ---------------------------------------------------- #
    def score_matrix(self, home: str, away: str, **boosts) -> np.ndarray:
        """(max_goals+1) x (max_goals+1) olasılık matrisi P[x, y]."""
        lam, mu = self.expected_goals(home, away, **boosts)
        n = self.max_goals + 1
        xs = np.arange(n)
        px = poisson.pmf(xs, lam)          # ev gol dağılımı
        py = poisson.pmf(xs, mu)           # deplasman gol dağılımı
        mat = np.outer(px, py)             # bağımsız Poisson
        # Dixon-Coles düzeltmesini 2x2 köşeye uygula
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        mat = mat * dc_tau(X, Y, lam, mu, self.rho)
        mat = np.clip(mat, 0, None)
        total = mat.sum()
        if total > 0:
            mat /= total                   # tau normalizasyonu bozduğu için tekrar normalle
        return mat

    # ----- tahmin ---------------------------------------------------------- #
    def predict(self, home: str, away: str, **boosts) -> dict:
        """Bir maç için tüm çıktı olasılıklarını üretir."""
        home = self.resolve_team(home)
        away = self.resolve_team(away)
        mat = self.score_matrix(home, away, **boosts)
        n = mat.shape[0]
        idx = np.arange(n)
        X, Y = np.meshgrid(idx, idx, indexing="ij")

        p_home = mat[X > Y].sum()
        p_draw = np.trace(mat)
        p_away = mat[X < Y].sum()

        p_btts_yes = mat[1:, 1:].sum()             # ev>=1 ve dep>=1
        p_btts_no = 1.0 - p_btts_yes

        totals = X + Y
        p_over25 = mat[totals >= 3].sum()
        p_under25 = mat[totals <= 2].sum()

        # en olası skor
        flat_idx = int(np.argmax(mat))
        best_x, best_y = flat_idx // n, flat_idx % n

        lam, mu = self.expected_goals(home, away, **boosts)
        return {
            "home": home,
            "away": away,
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
            "prob_btts_yes": float(p_btts_yes),
            "prob_btts_no": float(p_btts_no),
            "prob_over25": float(p_over25),
            "prob_under25": float(p_under25),
            "most_likely_score": (best_x, best_y),
            "exp_home_goals": float(lam),
            "exp_away_goals": float(mu),
            "score_matrix": mat,
        }


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def _neg_log_likelihood(params, home_idx, away_idx, hg, ag, weights, n_teams):
    """Ağırlıklı negatif log-olabilirlik (minimize edilecek amaç fonksiyonu)."""
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    lam = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] - defense[home_idx])

    tau = dc_tau(hg, ag, lam, mu, rho)
    # tau <= 0 olursa log tanımsız; küçük pozitif tabana kırp (ceza etkisi)
    tau = np.clip(tau, 1e-10, None)

    # Poisson log-pmf (sabit -log(x!) terimi optimizasyonu etkilemez ama
    # doğru log-likelihood için dahil ediyoruz)
    ll_home = -lam + hg * np.log(lam) - _log_factorial(hg)
    ll_away = -mu + ag * np.log(mu) - _log_factorial(ag)
    ll = weights * (np.log(tau) + ll_home + ll_away)
    return -np.sum(ll)


_LOGFACT_CACHE: dict[int, float] = {}


def _log_factorial(k):
    """Vektörize log(k!) — scipy.special.gammaln eşdeğeri, ek bağımlılıksız."""
    k = np.asarray(k, dtype=float)
    from scipy.special import gammaln
    return gammaln(k + 1.0)


def fit(
    matches: pd.DataFrame,
    ref_date: date | None = None,
    half_life: float = config.DEFAULT_HALF_LIFE_DAYS,
    max_goals: int = config.DEFAULT_MAX_GOALS,
    min_matches: int = 30,
) -> DixonColesModel:
    """Maç DataFrame'inden Dixon-Coles modelini fit eder.

    Args:
        matches: en az [date, home_team, away_team, home_goals, away_goals] sütunları.
        ref_date: zaman ağırlığı referansı (varsayılan: en son maç tarihi).
        half_life: zaman ağırlığı yarı-ömrü (gün). 0 => ağırlıksız.

    Raises:
        ValueError: yeterli maç yoksa.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    df = df[df["home_team"] != df["away_team"]]
    if len(df) < min_matches:
        raise ValueError(
            f"Model için yetersiz maç: {len(df)} < {min_matches}. "
            "Daha fazla sezon indirin."
        )

    df["date"] = pd.to_datetime(df["date"])
    if ref_date is None:
        ref_date = df["date"].max().date()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = df["home_team"].map(team_idx).to_numpy()
    away_idx = df["away_team"].map(team_idx).to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)
    weights = time_weights(df["date"], ref_date, half_life)

    # Başlangıç: attack/defense ~ 0, home_adv ~ 0.25, rho ~ -0.1
    x0 = np.concatenate([
        np.zeros(n),          # attack
        np.zeros(n),          # defense
        [0.25],               # home_adv
        [-0.1],               # rho
    ])
    # rho'yu makul aralıkta tut; diğerleri serbest
    bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1.0, 1.0)] + [(-0.2, 0.2)]

    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(home_idx, away_idx, hg, ag, weights, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-9},
    )

    attack = res.x[:n].copy()
    defense = res.x[n:2 * n].copy()
    home_adv = float(res.x[2 * n])
    rho = float(res.x[2 * n + 1])

    # Konum belirsizliğini kır: attack ortalamasını 0'a çek, farkı defense'e taşı
    # (attack_i - defense_j değişmez, yorumlanabilirlik artar)
    shift = attack.mean()
    attack -= shift
    defense -= shift

    return DixonColesModel(
        teams=teams,
        attack={t: float(attack[i]) for t, i in team_idx.items()},
        defense={t: float(defense[i]) for t, i in team_idx.items()},
        home_adv=home_adv,
        rho=rho,
        max_goals=max_goals,
        log_likelihood=float(-res.fun),
        n_matches=len(df),
    )
