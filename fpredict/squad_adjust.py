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


def fetch_injuries_for_team(
    api_key: str,
    league_key: str,
    team_name: str,
    season: int,
) -> list[dict]:
    """Takım ADINDAN yola çıkarak sakat/cezalı oyuncuları çeker.

    Önceki sürümde kullanıcıdan sayısal takım id'si bekleniyordu ve bu yüzden
    özellik arayüzde kullanılamıyordu. Artık takım id'si lig kadrosundan
    otomatik çözülür (isim eşleştirmesiyle).

    Returns:
        [{"player": str, "type": str, "reason": str}, ...]

    Raises:
        SquadAPIError: anahtar/lig id yoksa, takım bulunamazsa veya API hata verirse.
    """
    from .api_football import APIFootballClient, APIFootballError

    league_id = load_league_id(league_key)
    if not league_id:
        raise SquadAPIError(
            f"{league_key} için API-Football lig id'si tanımlı değil. "
            "Veri ekranındaki 'Lig ID ara' aracıyla belirleyebilirsiniz."
        )
    try:
        client = APIFootballClient(api_key, provider=load_api_provider())
        team_id = client.find_team_id(league_id, season, team_name)
        if not team_id:
            raise SquadAPIError(
                f"'{team_name}' API kadro listesinde bulunamadı "
                f"(lig id {league_id}, sezon {season})."
            )
        return client.fetch_injuries(team_id, season)
    except APIFootballError as exc:
        raise SquadAPIError(str(exc)) from exc


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


def save_api_provider(provider: str) -> None:
    """Kullanılacak API kanalını kaydeder ('direct' veya 'rapidapi')."""
    config.ensure_app_dir()
    settings = _load_settings()
    settings["apifootball_provider"] = provider
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def load_api_provider() -> str:
    """Kayıtlı API kanalı (varsayılan: doğrudan api-sports.io)."""
    return _load_settings().get("apifootball_provider") or "direct"


def save_league_id(league_key: str, league_id: int) -> None:
    """Kullanıcının doğruladığı API-Football lig id'sini kalıcı kaydeder."""
    config.ensure_app_dir()
    settings = _load_settings()
    ids = settings.setdefault("apifootball_league_ids", {})
    ids[league_key] = int(league_id)
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def load_league_id(league_key: str) -> Optional[int]:
    """Bu lig için kullanılacak API id'si.

    Öncelik kullanıcının kaydettiği değerdedir; yoksa yerleşik varsayılan
    kullanılır. Böylece yerleşik id yanlış çıkarsa kullanıcı düzeltebilir.
    """
    override = _load_settings().get("apifootball_league_ids", {}).get(league_key)
    if override:
        return int(override)
    return config.APIFOOTBALL_LEAGUE_IDS.get(league_key)


def _load_settings() -> dict:
    if config.SETTINGS_PATH.exists():
        try:
            return json.loads(config.SETTINGS_PATH.read_text())
        except (ValueError, OSError):
            return {}
    return {}
