"""Backtest ekranı: walk-forward test, metrikler ve grafik."""
from __future__ import annotations

import streamlit as st

from fpredict import backtest, config
from . import common


def render():
    st.header("📊 Backtest (Geçmişe Dönük Test)")
    st.caption(
        "Her maç, YALNIZCA kendisinden önceki maçlarla eğitilmiş modelle tahmin "
        "edilir (veri sızıntısı yok). Böylece modelin gerçek öngörü gücü ölçülür."
    )

    bust = st.session_state.get("data_version", 0)
    from fpredict import db
    summary = db.league_summary()
    if summary.empty:
        st.info("Önce **Veri** sekmesinden lig indirin.")
        return

    name_map = config.LEAGUE_NAMES
    league_key = st.selectbox(
        "Lig", options=list(summary["league"]),
        format_func=lambda c: f"{name_map.get(c, c)} ({c})",
    )

    c1, c2, c3 = st.columns(3)
    half_life = c1.slider("Yarı-ömür (gün)", 30, 720, int(config.DEFAULT_HALF_LIFE_DAYS), 30)
    min_train = c2.number_input("Isınma (ilk N maç)", 100, 2000, 200, 50)
    refit_every = c3.number_input("Kaç maçta bir yeniden fit", 5, 100, 20, 5)

    st.caption(
        "Not: Backtest her yeniden-fit'te modeli baştan çözer; büyük veri + sık "
        "refit yavaş olabilir. Refit aralığını artırmak hızlandırır."
    )

    _batch_calibration_panel(list(summary["league"]), half_life,
                             int(min_train), int(refit_every))

    if not st.button("▶️ Backtest'i Çalıştır", type="primary"):
        return

    df = common.cached_matches(league_key, bust)
    prog = st.progress(0.0, text="Çalışıyor…")
    try:
        res = backtest.run_backtest(
            df, half_life=half_life, min_train=int(min_train),
            refit_every=int(refit_every), progress=lambda f: prog.progress(min(f, 1.0)),
        )
    except ValueError as exc:
        prog.empty()
        st.error(str(exc))
        return
    prog.empty()

    # --- Metrikler ---------------------------------------------------------- #
    st.subheader("Sonuçlar")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tahmin edilen maç", res.n_predicted)
    m2.metric("Doğruluk", f"%{res.accuracy*100:.1f}",
              delta=f"{(res.accuracy-res.baseline_home_acc)*100:+.1f} pp vs ev-sahibi")
    m3.metric("Log-loss", f"{res.log_loss:.3f}",
              delta=f"{res.log_loss-res.baseline_uniform_logloss:+.3f} vs 1/3",
              delta_color="inverse")
    m4.metric("Brier", f"{res.brier:.3f}")

    st.info(res.commentary())

    # --- Karşılaştırma grafiği --------------------------------------------- #
    _plot_comparison(res)

    # --- Kalibrasyon / kümülatif doğruluk ---------------------------------- #
    _plot_cumulative(res)

    # --- Kalibrasyon -------------------------------------------------------- #
    _calibration_panel(res, league_key)

    with st.expander("Tahmin kayıtları (ilk 200)"):
        st.dataframe(res.records.head(200), use_container_width=True, hide_index=True)


def _batch_calibration_panel(leagues: list[str], half_life, min_train, refit_every):
    """Cache'teki tüm ligleri tek seferde kalibre eder.

    Kalibrasyon lig bazındadır (ev sahibi avantajı, beraberlik oranı ve gol
    dağılımı ligden lige değişir), ancak bunu elle tek tek yapmak zahmetlidir.
    """
    from fpredict import backtest, calibration, db

    with st.expander(f"⚙️ Tüm ligleri toplu kalibre et ({len(leagues)} lig)"):
        st.caption(
            "Her lig için backtest çalıştırıp kalibrasyonu öğrenir ve kaydeder. "
            "Yukarıdaki parametreler (yarı-ömür, ısınma, refit aralığı) kullanılır.\n\n"
            "⏱️ **Lig başına birkaç dakika sürebilir** — büyük cache'lerde uzun sürer. "
            "Veri yetersiz olan ligler atlanır."
        )

        # Hangi liglerde kalibrasyon zaten var?
        existing = {lg: calibration.load(lg) for lg in leagues}
        have = [lg for lg, c in existing.items() if c]
        if have:
            st.caption(f"Kayıtlı kalibrasyonu olanlar: {', '.join(have)}")

        skip_done = st.checkbox("Kalibrasyonu olan ligleri atla", value=True)

        if not st.button("⚙️ Toplu Kalibrasyonu Başlat"):
            return

        targets = [lg for lg in leagues if not (skip_done and existing.get(lg))]
        if not targets:
            st.info("Kalibre edilecek lig kalmadı.")
            return

        prog = st.progress(0.0, text="Başlatılıyor…")
        rows = []
        for i, lg in enumerate(targets):
            name = config.LEAGUE_NAMES.get(lg, lg)
            prog.progress(i / len(targets), text=f"{name} — backtest çalışıyor…")
            try:
                df = db.load_matches(lg)
                res = backtest.run_backtest(
                    df, half_life=half_life, min_train=min_train,
                    refit_every=refit_every,
                )
                cal = calibration.fit_from_backtest(res.records, lg)
                calibration.save(cal)
                rows.append({
                    "lig": name,
                    "maç": res.n_predicted,
                    "doğruluk %": round(res.accuracy * 100, 1),
                    "T (1X2)": round(cal.t_1x2, 2),
                    "log-loss": round(cal.logloss_before, 4),
                    "kalibre log-loss": round(cal.logloss_after, 4),
                    "durum": "✅ kaydedildi",
                })
            except ValueError as exc:
                rows.append({"lig": name, "maç": 0, "durum": f"⏭️ atlandı — {exc}"})
            except Exception as exc:  # beklenmeyen; diğer ligler devam etsin
                rows.append({"lig": name, "maç": 0, "durum": f"⚠️ hata — {exc}"})

        prog.empty()
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ok = sum(1 for r in rows if r.get("durum", "").startswith("✅"))
        st.success(
            f"{ok}/{len(targets)} lig kalibre edildi ve kaydedildi. "
            "**Ana (Tahmin)** ekranındaki olasılıklar artık kalibre gösterilecek."
        )
        common.bump_data_version()


def _calibration_panel(res, league_key: str):
    """Backtest sonuçlarından kalibrasyon öğrenme ve kaydetme."""
    from fpredict import calibration

    st.divider()
    st.subheader("🎚️ Olasılık Kalibrasyonu")
    st.caption(
        "Model, yüksek olasılıklarda fazla iddialı olabilir (ör. '%76' dediği "
        "maçların gerçekte %68'i tutar). Kalibrasyon, bu sapmayı backtest "
        "kayıtlarından öğrenip gösterilen yüzdeleri gerçekleşme oranına "
        "yaklaştırır. Tahminlerin **sıralaması değişmez**, yalnızca güven düzeyi "
        "düzeltilir."
    )

    existing = calibration.load(league_key)
    if existing:
        st.info(f"Bu ligde kayıtlı kalibrasyon var — {existing.describe()}")

    # Mevcut durum tablosu
    before = calibration.reliability_table(res.records, 1.0)
    if not before:
        st.caption("Güvenilirlik tablosu için yeterli kayıt yok.")
        return

    try:
        fitted = calibration.fit_from_backtest(res.records, league_key)
    except ValueError as exc:
        st.warning(f"Kalibrasyon öğrenilemedi: {exc}")
        return

    import pandas as pd

    comp = calibration.calibration_comparison(res.records, fitted.t_1x2)
    tbl = pd.DataFrame(comp).rename(columns={
        "ham_tahmin": "ham tahmin %",
        "kalibre_tahmin": "kalibre tahmin %",
        "gerçekleşme": "gerçekleşme %",
        "ham_fark": "ham fark",
        "kalibre_fark": "kalibre fark",
    })

    st.markdown(
        "**Model güveni vs gerçekleşme** — bantlar ham olasılığa göre sabittir, "
        "yani her satır **aynı maçları** gösterir. Sıcaklık ölçekleme sıralamayı "
        "değiştirmediği için gerçekleşme oranı iki durumda da aynıdır; tek soru "
        "gösterilen yüzdenin gerçeğe yaklaşıp yaklaşmadığıdır (**fark küçüldü mü?**)."
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Küçük örneklem uyarısı — dar bantlarda oranlar gürültülüdür
    small = [r["bant"] for r in comp if r["maç"] < 50]
    if small:
        st.caption(
            f"⚠️ Şu bantlarda 50'den az maç var: {', '.join(small)}. "
            "Bu bantlardaki gerçekleşme oranları gürültülüdür; tek başına "
            "yorumlamayın — asıl ölçüt aşağıdaki log-loss'tur."
        )

    improved = sum(1 for r in comp if r["kalibre_fark"] < r["ham_fark"])
    st.caption(
        f"{improved}/{len(comp)} bantta gösterilen yüzde gerçekleşmeye yaklaştı."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Öğrenilen sıcaklık (1X2)", f"{fitted.t_1x2:.2f}",
              help="1.0 = değişiklik yok. >1 = güven azaltılıyor.")
    c2.metric("Log-loss (kalibrasyonsuz)", f"{fitted.logloss_before:.4f}")
    c3.metric("Log-loss (kalibre)", f"{fitted.logloss_after:.4f}",
              delta=f"{fitted.logloss_after - fitted.logloss_before:+.4f}",
              delta_color="inverse")

    if fitted.is_identity:
        st.success(
            "Model bu ligde zaten iyi kalibre görünüyor — kaydetmeye gerek yok."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Kalibrasyonu Kaydet ve Uygula", type="primary"):
            calibration.save(fitted)
            st.success(
                "Kaydedildi. **Ana (Tahmin)** ekranındaki olasılıklar artık "
                "kalibre edilmiş olarak gösterilecek."
            )
    with col_b:
        if existing and st.button("🗑️ Kayıtlı kalibrasyonu sil"):
            calibration.clear(league_key)
            st.success("Silindi. Tahminler ham model çıktısına döndü.")
            st.rerun()


def _plot_comparison(res):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    # log-loss karşılaştırması
    labels = ["Model", "Naif ev-sahibi", "Bilgisiz 1/3"]
    vals = [res.log_loss, res.baseline_home_logloss, res.baseline_uniform_logloss]
    axes[0].bar(labels, vals, color=["#2e7d32", "#9e9e9e", "#bdbdbd"])
    axes[0].set_title("Log-loss (düşük = iyi)")
    axes[0].tick_params(axis="x", rotation=15)

    accs = [res.accuracy, res.baseline_home_acc]
    axes[1].bar(["Model", "Hep ev-sahibi"], accs, color=["#1565c0", "#9e9e9e"])
    axes[1].set_title("Doğruluk (yüksek = iyi)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig)


def _plot_cumulative(res):
    import matplotlib.pyplot as plt
    import numpy as np

    if res.records.empty:
        return
    rec = res.records.copy()
    correct = (rec["pred"] == rec["actual"]).to_numpy(dtype=float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(rec["date"], cum_acc, color="#1565c0", label="Model kümülatif doğruluk")
    ax.axhline(res.baseline_home_acc, color="#9e9e9e", ls="--", label="Hep ev-sahibi")
    ax.set_ylabel("Kümülatif doğruluk")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
