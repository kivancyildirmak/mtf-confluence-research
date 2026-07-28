"""Ekranlar arası paylaşılan yardımcılar: model cache, takım listesi vb."""
from __future__ import annotations

import streamlit as st

from fpredict import db, model


@st.cache_data(show_spinner=False)
def cached_matches(league_key: str, _cache_bust: float):
    """Cache'deki maçları yükler. `_cache_bust` değişince yeniden okunur."""
    return db.load_matches(league_key)


@st.cache_resource(show_spinner="Model fit ediliyor…")
def cached_model(league_key: str, half_life: float, n_matches: int, _bust: float):
    """Bir lig için Dixon-Coles modelini fit edip önbelleğe alır.

    `n_matches` ve `_bust`, cache anahtarının parçasıdır; veri güncellenince
    (maç sayısı değişince) model otomatik yeniden fit edilir.
    """
    matches = db.load_matches(league_key)
    return model.fit(matches, half_life=half_life)


def teams_for_league(league_key: str, bust: float) -> list[str]:
    df = cached_matches(league_key, bust)
    if df.empty:
        return []
    return sorted(set(df["home_team"]) | set(df["away_team"]))


def bump_data_version():
    """Veri güncellenince cache'leri geçersiz kılmak için sürüm sayacını artırır."""
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    cached_matches.clear()
    cached_model.clear()
