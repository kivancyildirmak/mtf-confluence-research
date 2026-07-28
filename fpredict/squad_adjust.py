"""Sakatlık / cezalı / kadro bilgisini modele 'güç çarpanı' olarak yansıtma.

İki kaynak desteklenir:

1. **API-Football (api-sports.io)** — kullanıcının girdiği API anahtarı ile
   `injuries` endpoint'inden bir maç/takım için sakat oyuncu listesi çekilir.
   Anahtar yoksa/başarısızsa sessizce elle moda düşülür.

2. **Elle giriş** — kullanıcı her takım için doğrudan bir "güç çarpanı" (0.5–1.2)
   girebilir veya eksik oyuncu sayısı üzerinden kaba bir çarpan hesaplatabilir.

ÖNEMLİ: Bu ayarlama kaba bir yaklaşımdır. Sakatlık verisini gerçek oyuncu
katkısına (xG, dakika, pozisyon) bağlamadan yapılan her düzeltme spekülatiftir;
arayüz bunu kullanıcıya açıkça belirtir.

Güç çarpanı, model.expected_goals içindeki home_boost/away_boost'a beslenir:
    - hücum çarpanı 1'in altındaysa takım daha az gol atar,
    - savunma için ayrı bir çarpan da rakibin beklenen golünü artırabilir.
Basitlik için tek bir "attack_mult" ve "defense_mult" tutarız.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

from . import config


# --------------------------------------------------------------------------- #
# Güç çarpanı veri modeli
# --------------------------------------------------------------------------- #
@dataclass
class SquadAdjustment:
    """Bir takım için hücum/savunma güç çarpanları (1.0 = etkisiz)."""
    attack_mult: float = 1.0
    defense_mult: float = 1.0   # >1 => savunma daha zayıf, rakip daha çok gol atar
    note: str = ""

    def clamped(self) -> "SquadAdjustment":
        """Çarpanları mantıklı aralığa sıkıştırır (aşırı değerleri engeller)."""
        return SquadAdjustment(
            attack_mult=min(max(self.attack_mult, 0.5), 1.2),
            defense_mult=min(max(self.defense_mult, 0.8), 1.5),
            note=self.note,
        )


def multiplier_from_missing(
    n_key_missing: int, per_player_penalty: float = 0.06
) -> float:
    """Eksik kilit oyuncu sayısından kaba bir hücum çarpanı üretir.

    Örn. 3 kilit oyuncu eksik, oyuncu başına %6 => 1 - 3*0.06 = 0.82.
    """
    return max(0.5, 1.0 - n_key_missing * per_player_penalty)


def apply_to_boosts(home_adj: SquadAdjustment, away_adj: SquadAdjustment) -> dict:
    """İki takımın ayarlarını model.predict için boost sözlüğüne çevirir.

    Ev sahibinin beklenen golü = kendi hücumu * kendi attack_mult, AMA rakibin
    savunması zayıfsa (away defense_mult>1) ev sahibi daha çok atar.
    """
    home_adj = home_adj.clamped()
    away_adj = away_adj.clamped()
    return {
        # ev golü: ev hücumu yukarı, deplasman savunması zayıflığı yukarı
        "home_boost": home_adj.attack_mult * away_adj.defense_mult,
        # deplasman golü: deplasman hücumu yukarı, ev savunması zayıflığı yukarı
        "away_boost": away_adj.attack_mult * home_adj.defense_mult,
    }


# --------------------------------------------------------------------------- #
# API-Football sakatlık çekimi (opsiyonel)
# --------------------------------------------------------------------------- #
class SquadAPIError(RuntimeError):
    """API çağrısı başarısız olduğunda kullanıcıya gösterilebilir hata."""


def fetch_injuries_api_football(
    api_key: str,
    team_id: int,
    season: int,
    timeout: int = config.HTTP_TIMEOUT,
) -> list[dict]:
    """API-Football injuries endpoint'inden takımın sakat oyuncularını çeker.

    Returns:
        [{"player": str, "type": str, "reason": str}, ...]

    Raises:
        SquadAPIError: anahtar yoksa veya ağ/HTTP hatası olursa.
    """
    if not api_key:
        raise SquadAPIError("API anahtarı girilmemiş.")
    import requests

    url = "https://v3.football.api-sports.io/injuries"
    try:
        resp = requests.get(
            url,
            params={"team": team_id, "season": season},
            headers={"x-apisports-key": api_key},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        raise SquadAPIError(f"Sakatlık API'sine erişilemedi: {exc}") from exc
    except ValueError as exc:
        raise SquadAPIError(f"API yanıtı çözümlenemedi: {exc}") from exc

    if data.get("errors"):
        raise SquadAPIError(f"API hatası: {data['errors']}")

    out = []
    for item in data.get("response", []):
        player = item.get("player", {}) or {}
        out.append({
            "player": player.get("name", "?"),
            "type": player.get("type", ""),
            "reason": player.get("reason", ""),
        })
    return out


# --------------------------------------------------------------------------- #
# Ayarların diske kaydı (settings.json)
# --------------------------------------------------------------------------- #
def save_api_key(key: str) -> None:
    config.ensure_app_dir()
    settings = _load_settings()
    settings["apifootball_key"] = key
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def load_api_key() -> Optional[str]:
    return _load_settings().get("apifootball_key")


def _load_settings() -> dict:
    if config.SETTINGS_PATH.exists():
        try:
            return json.loads(config.SETTINGS_PATH.read_text())
        except (ValueError, OSError):
            return {}
    return {}
