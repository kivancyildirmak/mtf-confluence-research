"""Veri ekranı: cache durumu, lig güncelleme, elle CSV yükleme (yedek)."""
from __future__ import annotations

import streamlit as st

from fpredict import config, data_fetch, db, squad_adjust
from . import common


def render():
    st.header("📥 Veri Yönetimi")
    st.caption(
        "Veriler [football-data.co.uk](https://www.football-data.co.uk/) üzerinden "
        "otomatik indirilir ve yerel SQLite cache'e yazılır. Her açılışta baştan "
        "indirilmez; yalnızca güncelleme butonuna basınca yeni maçlar eklenir."
    )

    # --- Cache özeti --------------------------------------------------------- #
    st.subheader("Cache Durumu")
    summary = db.league_summary()
    if summary.empty:
        st.info("Cache boş. Aşağıdan bir lig seçip **Verileri Güncelle**'ye basın.")
    else:
        # okunur lig adları ekle
        name_map = config.LEAGUE_NAMES
        summary.insert(1, "lig_adı", summary["league"].map(name_map).fillna(summary["league"]))
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # --- Güncelleme --------------------------------------------------------- #
    st.subheader("Verileri Güncelle")
    col1, col2 = st.columns([2, 1])
    with col1:
        league_key = st.selectbox(
            "Lig",
            options=list(config.LEAGUES.keys()),
            format_func=config.league_label,
        )
    is_extra = config.LEAGUES[league_key].source == "extra"
    with col2:
        seasons_back = st.number_input(
            "Kaç sezon", min_value=1, max_value=15,
            value=config.DEFAULT_SEASONS_BACK,
            disabled=is_extra,
            help=("Bu lig tek dosyada tüm sezonlarla geldiği için sezon sayısı "
                  "seçilemez; tam geçmiş indirilir." if is_extra else
                  "Kaç sezon geriye gidilsin."),
        )
    if is_extra:
        st.caption(
            "ℹ️ Bu lig football-data.co.uk'ta **tek dosyada tüm sezonlar** biçiminde "
            "yayınlanır; indirme tüm geçmişi getirir (model zaten son maçlara "
            "daha fazla ağırlık verir)."
        )

    # --- Kaynak seçimi ------------------------------------------------------ #
    in_archive = config.archive_division(league_key) is not None
    api_key = squad_adjust.load_api_key()

    options = ["Otomatik (önerilen)"]
    if in_archive:
        options.append("Arşiv (GitHub) — güncel değil")
    options.append("API-Football — güncel")

    source_mode = st.radio(
        "Veri kaynağı",
        options,
        horizontal=True,
        help=(
            "Otomatik: önce football-data.co.uk, olmazsa GitHub aynası.\n"
            "Arşiv: engel varken tüm ligler için çalışır ama veri eskidir.\n"
            "API-Football: güncel sezon dahil; ücretsiz anahtar gerekir."
        ),
    )
    use_archive = source_mode.startswith("Arşiv")
    use_api = source_mode.startswith("API-Football")

    if use_api:
        _api_panel(league_key, api_key)
        if not api_key:
            st.stop()

    if st.button("🔄 Verileri Güncelle", type="primary"):
        prog = st.progress(0.0, text="Başlatılıyor…")

        def _cb(msg, frac):
            prog.progress(min(frac, 1.0), text=msg)

        try:
            if use_archive:
                result = data_fetch.update_from_archive(league_key, progress=_cb)
            elif use_api:
                result = data_fetch.update_from_api(
                    league_key, api_key, seasons_back=int(seasons_back), progress=_cb
                )
                if result.get("quota"):
                    st.caption(f"API kotası: {result['quota']}")
            else:
                result = data_fetch.update_league(
                    league_key, seasons_back=int(seasons_back), progress=_cb
                )
            common.bump_data_version()
            prog.empty()
            if result["seasons_ok"]:
                srcs = result.get("sources_used") or []
                st.success(
                    f"Güncellendi: {result['inserted']} maç işlendi. "
                    f"Başarılı sezonlar: {', '.join(result['seasons_ok'])}."
                    + (f" (Kaynak: {', '.join(srcs)})" if srcs else "")
                )
                if "GitHub aynası" in srcs:
                    st.info(
                        "ℹ️ football-data.co.uk'a erişilemediği için **yedek ayna** "
                        "kullanıldı. Sonuç/skor verisi eksiksizdir, ancak aynada "
                        "**bahis oranı sütunları yoktur** — bu yüzden *Değer* ekranı "
                        "bu lig için çalışmaz. Tahmin ve Backtest normal çalışır."
                    )
                _staleness_warning(result)
            if result["seasons_failed"]:
                # Hiçbir sezon inmediyse bu bir ağ sorunudur: teşhis göster
                if not result["seasons_ok"]:
                    _network_diagnosis(result["errors"], league_key)
                else:
                    st.warning(
                        "Bazı sezonlar indirilemedi: "
                        f"{', '.join(result['seasons_failed'])}. "
                        "(En yeni sezon henüz yayınlanmamış olabilir — bu normaldir.)"
                    )
                    with st.expander("Ayrıntılı hata"):
                        for e in result["errors"]:
                            st.caption(f"• {e}")
                _manual_upload_fallback(league_key)
        except data_fetch.DataFetchError as exc:
            prog.empty()
            st.error(f"İndirme hatası: {exc}")
            _manual_upload_fallback(league_key)
        except Exception as exc:  # beklenmeyen
            prog.empty()
            st.error(f"Beklenmeyen hata: {exc}")

    st.divider()
    with st.expander("📄 Elle CSV Yükle (yedek yöntem)"):
        _manual_upload_fallback(league_key, standalone=True)


def _api_panel(league_key: str, api_key: str | None):
    """API-Football kanalı, anahtarı, bağlantı testi ve lig ID doğrulama aracı."""
    from fpredict.api_football import PROVIDERS

    # --- Kanal seçimi ------------------------------------------------------- #
    keys = list(PROVIDERS.keys())
    current = squad_adjust.load_api_provider()
    provider = st.radio(
        "API kanalı",
        keys,
        index=keys.index(current) if current in keys else 0,
        format_func=lambda k: PROVIDERS[k]["label"],
        horizontal=True,
        help=(
            "Aynı API iki farklı alan adından sunulur. Biri ağınızda "
            "engelliyse diğerini deneyin — veri ve kullanım aynıdır."
        ),
    )
    if provider != current:
        squad_adjust.save_api_provider(provider)
        st.caption(f"Kanal '{PROVIDERS[provider]['label']}' olarak kaydedildi.")

    if not api_key:
        st.warning(
            f"Bu kaynak için ücretsiz bir **API anahtarı** gerekir "
            f"(günde 100 istek).\n\n"
            f"1. Kayıt olun: {PROVIDERS[provider]['signup']}\n"
            "2. Panelden anahtarınızı kopyalayın.\n"
            "3. Aşağıya yapıştırıp kaydedin."
        )
        st.caption(
            "Not: RapidAPI anahtarı ile doğrudan api-sports.io anahtarı "
            "**farklıdır**; seçtiğiniz kanalın anahtarını girin."
        )
        new_key = st.text_input("API anahtarı", type="password", key="api_key_data")
        if st.button("💾 Anahtarı Kaydet"):
            if new_key.strip():
                squad_adjust.save_api_key(new_key.strip())
                st.success("Anahtar kaydedildi. Sayfayı yenileyin.")
                st.rerun()
            else:
                st.error("Anahtar boş olamaz.")
        return

    with st.expander("🔑 Anahtarı değiştir"):
        new_key = st.text_input("Yeni API anahtarı", type="password", key="api_key_edit")
        if st.button("Kaydet", key="save_key_edit") and new_key.strip():
            squad_adjust.save_api_key(new_key.strip())
            st.success("Anahtar güncellendi.")
            st.rerun()

    league_id = squad_adjust.load_league_id(league_key)
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🔌 Bağlantıyı Test Et"):
            _test_api_connection(api_key, provider)
    with c2:
        st.caption(
            f"Kullanılacak lig ID: **{league_id}** — "
            "sonuç boş gelirse aşağıdaki araçla doğrulayın."
        )

    with st.expander("🔎 Lig ID ara / düzelt"):
        st.caption(
            "Yerleşik ID yanlışsa (API 0 maç döndürüyorsa) ligi adıyla arayıp "
            "doğru ID'yi seçebilirsiniz. Seçiminiz kalıcı kaydedilir."
        )
        query = st.text_input("Lig adı (en az 3 harf)", value="", key="lg_search")
        if st.button("Ara") and query.strip():
            try:
                from fpredict.api_football import APIFootballClient
                results = APIFootballClient(
                    api_key, provider=squad_adjust.load_api_provider()
                ).search_leagues(query.strip())
                if not results:
                    st.info("Sonuç bulunamadı.")
                else:
                    st.session_state["league_search_results"] = results
            except Exception as exc:
                st.error(f"Arama başarısız: {exc}")

        for r in st.session_state.get("league_search_results", [])[:12]:
            cols = st.columns([4, 2, 2, 2])
            cols[0].write(f"**{r['name']}** ({r['country']})")
            cols[1].write(f"tip: {r.get('type', '-')}")
            cols[2].write(f"ID: `{r['id']}`")
            if cols[3].button("Bunu kullan", key=f"pick_{r['id']}"):
                squad_adjust.save_league_id(league_key, r["id"])
                st.success(f"{league_key} için lig ID {r['id']} kaydedildi.")
                st.rerun()


def _test_api_connection(api_key: str, provider: str = "direct"):
    """Anahtarı /status ile doğrular ve kotayı gösterir."""
    try:
        from fpredict.api_football import APIFootballClient
        info = APIFootballClient(api_key, provider=provider).status()
    except Exception as exc:
        st.error(f"Bağlantı başarısız: {exc}")
        return
    quota = info.get("quota")
    st.success(
        f"✅ Bağlantı çalışıyor. Plan: **{info.get('plan') or '-'}**"
        + (f" — {quota.describe()}" if quota else "")
    )


def _staleness_warning(result: dict):
    """Veri güncel değilse kullanıcıyı açıkça uyarır.

    Arşiv kaynağı canlı değildir; kaç ay geride olduğunu ve bunun tahmine
    etkisini saklamadan söylemek gerekir.
    """
    stale = result.get("stale_days")
    last = result.get("last_match_date")
    if stale is None or last is None:
        return
    months = stale / 30.4
    if months < 3:
        st.caption(f"Son maç: {last} (veri güncel).")
        return

    st.warning(
        f"⚠️ **Veri güncel değil.** Cache'teki son maç: **{last}** "
        f"(yaklaşık **{months:.0f} ay** önce).\n\n"
        "Model bu tarihten sonraki transferleri, yükselen/düşen takımları ve "
        "form değişimlerini **görmez**. Yeni sezon maçlarında tahminlerin "
        "güvenilirliği belirgin biçimde düşer.\n\n"
        "Güncel veri için: football-data.co.uk erişimini açın (DNS/VPN) veya "
        "*Kadro/Sakatlık* ekranından API-Football anahtarı girin."
    )


def _network_diagnosis(errors: list[str], league_key: str):
    """Tüm indirmeler başarısızsa: olası sebep + somut çözüm adımları."""
    blob = " ".join(errors).lower()
    st.error("Hiçbir sezon indirilemedi — veri kaynağına ulaşılamıyor.")

    if "tls" in blob or "sertifika" in blob or "certificate" in blob:
        likely = (
            "**TLS araya girme tespit edildi.** Bağlantı, sunucunun kendi "
            "sertifikası yerine başka bir sertifikayla karşılanıyor. Bu tipik "
            "olarak İSS/DNS seviyesinde engelleme veya antivirüsün HTTPS "
            "taraması anlamına gelir."
        )
    elif "zaman aşımı" in blob or "reset" in blob or "kapatıldı" in blob:
        likely = (
            "**Bağlantı engelleniyor.** Paketler sunucuya gidiyor ancak yanıt "
            "dönmüyor veya bağlantı zorla kapatılıyor — ağ seviyesinde "
            "filtreleme belirtisi."
        )
    else:
        likely = "**Ağ hatası.** İnternet bağlantınızda bir sorun olabilir."

    st.markdown(f"### Olası sebep\n{likely}")
    st.markdown(
        """
### Deneyebilecekleriniz

1. **Tarayıcıda test edin:** `https://www.football-data.co.uk/englandm.php`
   adresini açın. Açılmıyorsa sorun uygulamada değil, ağ erişimindedir.
2. **DNS değiştirin** (engellerin çoğu DNS seviyesindedir):
   Ayarlar → Ağ → Bağdaştırıcı → IPv4 → DNS: `1.1.1.1` ve `8.8.8.8`.
   Sonra komut satırında `ipconfig /flushdns` çalıştırıp tekrar deneyin.
3. **Antivirüsün HTTPS/SSL taramasını** geçici kapatın (Kaspersky, ESET,
   Avast vb. bu hatayı üretebilir).
4. **VPN** kullanın.
5. **Elle CSV yükleyin:** Dosyayı başka bir cihazdan/ağdan indirip aşağıdan
   yükleyebilirsiniz.
        """
    )
    if config.has_mirror(config.LEAGUES[league_key].code):
        st.info("Bu lig için yedek ayna da denendi ve o da başarısız oldu.")
    else:
        st.info(
            f"**Not:** {config.LEAGUES[league_key].name} için yedek ayna yok. "
            "Ayna yalnızca Premier Lig, La Liga, Serie A, Bundesliga ve Ligue 1'i "
            "kapsar — bu ligleri engelden bağımsız indirebilirsiniz."
        )

    with st.expander("Teknik hata ayrıntısı"):
        for e in errors:
            st.caption(f"• {e}")


def _manual_upload_fallback(league_key: str, standalone: bool = False):
    """İndirme başarısızsa kullanıcı football-data CSV'sini elle yükleyebilir."""
    if not standalone:
        st.markdown("**İnternet erişimi yoksa:** CSV'yi elle yükleyebilirsiniz.")
    code = config.LEAGUES[league_key].code
    up = st.file_uploader(
        f"{config.LEAGUES[league_key].name} için football-data.co.uk CSV'si",
        type=["csv"], key=f"upload_{league_key}_{standalone}",
    )
    if up is not None:
        try:
            # Biçim otomatik algılanır (HomeTeam/FTHG veya Home/HG)
            rows = data_fetch.parse_any_csv_bytes(up.getvalue(), code)
            n = db.upsert_matches(rows)
            db.touch_updated(code)
            common.bump_data_version()
            st.success(f"{n} maç yüklendi ve cache'e eklendi.")
        except data_fetch.DataFetchError as exc:
            st.error(f"Dosya işlenemedi: {exc}")
