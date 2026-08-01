"""Model katmanı: LightGBM eğitimi, SHAP açıklanabilirliği, kaydet/yükle.

Neden LightGBM? Finansal özellikler ölçeksiz, gürültülü, eksik değer dolu ve
doğrusal olmayan etkileşimler içerir. Gradient boosting bunların hepsini
ölçekleme/imputasyon gerektirmeden işler; NaN'ı doğal olarak bir "yön" olarak
öğrenir (mikroyapı sütunları yokken bu kritik).

Model çıktısı bilinçli olarak **olasılıktır** (sınıf değil): :mod:`sizing`
olasılığı doğrudan pozisyon büyüklüğüne çevirir, :mod:`backtest` ise eşik ve
beklenen kazanç hesabında kullanır.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CONFIG, ModelConfig, ResearchConfig

#: Kaydedilen model dosyasının biçim sürümü (geriye uyumluluk kontrolü için).
MODEL_FORMAT_VERSION: str = "1.0"


@dataclass
class TrainedModel:
    """Eğitilmiş modeli ve onu yeniden üretmek için gereken HER ŞEYİ taşır.

    Modelin tek başına saklanması yetmez: canlı bot, özelliklerin hangi
    parametrelerle üretildiğini ve sütun sırasını bilmek zorundadır. Bu sınıf
    ikisini bir arada tutarak feature parity'yi denetlenebilir kılar.

    Attributes:
        booster: LightGBM modeli.
        feature_names: Eğitimde kullanılan sütunlar (SIRALI).
        config: Eğitim anındaki tam konfigürasyon anlık görüntüsü.
        metadata: Eğitim tarihi, örnek sayısı, metrikler vb.
    """

    booster: lgb.Booster
    feature_names: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Pozitif sınıf olasılığını döndürür.

        Args:
            X: Özellik matrisi. Sütunlar eğitimdekiyle AYNI olmalıdır.

        Returns:
            ``[0, 1]`` aralığında olasılık dizisi.

        Raises:
            ValueError: Sütun şeması eğitimdekiyle uyuşmuyorsa (feature parity
                ihlali — canlıda sessizce yanlış tahmin üretmesindense burada
                patlaması iyidir).
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"FEATURE PARITY İHLALİ — modelin beklediği sütunlar eksik: {missing[:10]}"
            )
        return np.asarray(
            self.booster.predict(X.loc[:, self.feature_names], raw_score=False),
            dtype="float64",
        )


# --------------------------------------------------------------------------- #
# Eğitim
# --------------------------------------------------------------------------- #


def train_model(
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    X_valid: pd.DataFrame | None = None,  # noqa: N803
    y_valid: pd.Series | None = None,
    valid_weight: pd.Series | np.ndarray | None = None,
    cfg: ModelConfig | None = None,
    full_config: ResearchConfig | None = None,
) -> TrainedModel:
    """LightGBM ikili sınıflandırıcı eğitir.

    Args:
        X_train: Eğitim özellikleri.
        y_train: Eğitim etiketleri (0/1).
        sample_weight: Örnek ağırlıkları (:func:`labeling.get_sample_weights`).
            Çakışan etiketlerin etkisini kırdığı için KULLANILMASI önerilir.
        X_valid: Erken durdurma için doğrulama özellikleri.
        y_valid: Doğrulama etiketleri.
        valid_weight: Doğrulama ağırlıkları.
        cfg: Model konfigürasyonu.
        full_config: Modelle birlikte saklanacak tam konfigürasyon.

    Returns:
        :class:`TrainedModel` örneği.

    Raises:
        ValueError: Eğitim kümesi boşsa veya tek sınıf içeriyorsa.

    Notes:
        SIZINTI: ``X_valid`` erken durdurma için kullanıldığında, doğrulama
        kümesi model seçimine dolaylı olarak katılır. Bu yüzden ``X_valid``
        CV'nin TEST katmanı OLMAMALIDIR; eğitim kümesinin sonundan ayrılmış
        (purge edilmiş) bir dilim olmalıdır.
    """
    c = cfg or CONFIG.model
    if len(X_train) == 0:
        raise ValueError("Eğitim kümesi boş.")
    y = np.asarray(y_train, dtype="float64")
    if len(np.unique(y[~np.isnan(y)])) < 2:
        raise ValueError("Eğitim etiketleri tek sınıf içeriyor; model eğitilemez.")

    feature_names = list(X_train.columns)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")

    dtrain = lgb.Dataset(
        X_train, label=y, weight=w, feature_name=feature_names, free_raw_data=False
    )

    params = dict(c.params)
    valid_sets: list[lgb.Dataset] = []
    valid_names: list[str] = []
    callbacks = [lgb.log_evaluation(period=0)]

    if X_valid is not None and y_valid is not None and len(X_valid) > 0:
        yv = np.asarray(y_valid, dtype="float64")
        if len(np.unique(yv)) < 2:
            # Tek sınıflı doğrulamada AUC tanımsızdır: erken durdurmayı kapat.
            warnings.warn("Doğrulama kümesi tek sınıf; erken durdurma atlandı.")
        else:
            vw = None if valid_weight is None else np.asarray(valid_weight, dtype="float64")
            valid_sets.append(
                lgb.Dataset(
                    X_valid.loc[:, feature_names],
                    label=yv,
                    weight=vw,
                    reference=dtrain,
                    free_raw_data=False,
                )
            )
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(c.early_stopping_rounds, verbose=False)
            )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=c.num_boost_round,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=callbacks,
    )

    metadata: dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_features": len(feature_names),
        "pozitif_oran": float(np.nanmean(y)),
        "best_iteration": int(booster.best_iteration or booster.num_trees()),
        "egitim_zamani": pd.Timestamp.utcnow().isoformat(),
        "format_version": MODEL_FORMAT_VERSION,
        "agirlik_kullanildi": w is not None,
    }
    if X_train.index.size:
        metadata["egitim_baslangic"] = str(X_train.index[0])
        metadata["egitim_bitis"] = str(X_train.index[-1])

    cfg_snapshot = (full_config or CONFIG).to_dict()
    return TrainedModel(
        booster=booster,
        feature_names=feature_names,
        config=cfg_snapshot,
        metadata=metadata,
    )


def predict_proba(model: TrainedModel, X: pd.DataFrame) -> pd.Series:  # noqa: N803
    """Model olasılıklarını, girdinin indeksini koruyarak döndürür.

    Args:
        model: Eğitilmiş model.
        X: Özellik matrisi.

    Returns:
        Olasılık serisi.
    """
    return pd.Series(model.predict_proba(X), index=X.index, name="prob")


# --------------------------------------------------------------------------- #
# Metrikler
# --------------------------------------------------------------------------- #


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    prob: pd.Series | np.ndarray,
    threshold: float = 0.5,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Sınıflandırma metriklerini hesaplar.

    **ACCURACY'E GÜVENME.** Etiketler dengesizse (ki finansal veride hep
    öyledir) accuracy anlamsızdır. Gerçek karar burada değil, maliyetli
    backtest'tedir (bkz. :mod:`backtest`).

    Args:
        y_true: Gerçek etiketler (0/1).
        prob: Model olasılıkları.
        threshold: Sınıflandırma eşiği.
        sample_weight: Metriklerde kullanılacak ağırlıklar (genelde ``None``:
            raporlanan skorun gerçek işlem dağılımını yansıtması için).

    Returns:
        Metrik sözlüğü.
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(prob, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype="float64")[mask]
    if len(y) == 0:
        return {"n": 0.0}

    pred = (p >= threshold).astype("int64")
    out: dict[str, float] = {
        "n": float(len(y)),
        "pozitif_oran": float(y.mean()),
        "tahmin_pozitif_oran": float(pred.mean()),
        "accuracy": float(accuracy_score(y, pred, sample_weight=sample_weight)),
        "precision": float(
            precision_score(y, pred, zero_division=0, sample_weight=sample_weight)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0, sample_weight=sample_weight)
        ),
        "f1": float(f1_score(y, pred, zero_division=0, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, p, sample_weight=sample_weight)),
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p, sample_weight=sample_weight))
        out["avg_precision"] = float(
            average_precision_score(y, p, sample_weight=sample_weight)
        )
        out["logloss"] = float(
            log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=sample_weight)
        )
    else:
        out["auc"] = float("nan")
        out["avg_precision"] = float("nan")
        out["logloss"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def compute_shap_values(
    model: TrainedModel,
    X: pd.DataFrame,  # noqa: N803
    max_samples: int | None = None,
    seed: int = CONFIG.seed,
) -> pd.DataFrame:
    """SHAP değerlerini hesaplar (TreeExplainer, tam ve hızlı).

    Args:
        model: Eğitilmiş model.
        X: Açıklanacak örnekler.
        max_samples: Hız için rastgele alt örnekleme boyutu.
        seed: Alt örnekleme tohumu (tekrarlanabilirlik).

    Returns:
        Örnek x özellik boyutunda SHAP değerleri tablosu.

    Notes:
        SHAP, LightGBM'in kendi ``gain`` önemine göre daha güvenilirdir: gain
        yüksek kardinaliteli özellikleri kayırır. Yine de SHAP da NEDENSELLİK
        göstermez — sadece modelin bu veride neye baktığını gösterir.
    """
    import shap  # yerel import: shap ağır bir bağımlılık

    n_max = max_samples if max_samples is not None else CONFIG.model.shap_max_samples
    Xs = X.loc[:, model.feature_names]
    if len(Xs) > n_max:
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(len(Xs), size=n_max, replace=False))
        Xs = Xs.iloc[pos]

    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(Xs)
    if isinstance(values, list):  # eski shap sürümleri sınıf başına liste döndürür
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, f, sinif)
        values = values[:, :, -1]
    return pd.DataFrame(values, index=Xs.index, columns=model.feature_names)


def feature_importance(
    model: TrainedModel,
    shap_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Özellik önemini SHAP ve LightGBM gain'i birlikte raporlar.

    Args:
        model: Eğitilmiş model.
        shap_values: :func:`compute_shap_values` çıktısı (opsiyonel).

    Returns:
        ``shap_mean_abs``, ``gain``, ``split`` sütunlu, öneme göre sıralı tablo.
    """
    gain = pd.Series(
        model.booster.feature_importance(importance_type="gain"),
        index=model.booster.feature_name(),
        name="gain",
    )
    split = pd.Series(
        model.booster.feature_importance(importance_type="split"),
        index=model.booster.feature_name(),
        name="split",
    )
    out = pd.concat([gain, split], axis=1)
    if shap_values is not None and not shap_values.empty:
        out["shap_mean_abs"] = shap_values.abs().mean()
        out = out.sort_values("shap_mean_abs", ascending=False)
    else:
        out["shap_mean_abs"] = np.nan
        out = out.sort_values("gain", ascending=False)
    return out.loc[:, ["shap_mean_abs", "gain", "split"]]


def plot_feature_importance(
    importance: pd.DataFrame,
    path: str | Path,
    top_n: int = 25,
    title: str = "Özellik Önemi (ortalama |SHAP|)",
) -> Path:
    """Özellik önemi grafiğini diske yazar.

    Args:
        importance: :func:`feature_importance` çıktısı.
        path: Hedef PNG yolu.
        top_n: Gösterilecek özellik sayısı.
        title: Grafik başlığı.

    Returns:
        Yazılan dosyanın yolu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = "shap_mean_abs" if importance["shap_mean_abs"].notna().any() else "gain"
    top = importance.sort_values(col, ascending=False).head(top_n).iloc[::-1]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.32 * len(top))))
    ax.barh(top.index, top[col].to_numpy(), color="#2b6cb0")
    ax.set_xlabel(col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Kaydet / yükle
# --------------------------------------------------------------------------- #


def save_model(model: TrainedModel, path: str | Path) -> Path:
    """Modeli TEK dosya olarak diske yazar (joblib).

    Dosya; booster'ı, özellik şemasını, config anlık görüntüsünü ve
    metaveriyi birlikte içerir. Canlı botun ihtiyacı olan her şey tek
    yerdedir.

    Args:
        model: Kaydedilecek model.
        path: Hedef dosya yolu (``.pkl`` önerilir).

    Returns:
        Yazılan dosyanın yolu.
    """
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "booster_str": model.booster.model_to_string(),
        "feature_names": model.feature_names,
        "config": model.config,
        "metadata": model.metadata,
    }
    joblib.dump(payload, p, compress=3)

    # İnsan tarafından okunabilir yan dosya (deney kaydı / denetim için).
    side = p.with_suffix(".meta.json")
    side.write_text(
        json.dumps(
            {"metadata": model.metadata, "feature_names": model.feature_names},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


def load_model(path: str | Path) -> TrainedModel:
    """:func:`save_model` ile yazılmış modeli geri yükler.

    Args:
        path: Model dosyası yolu.

    Returns:
        :class:`TrainedModel` örneği.

    Raises:
        FileNotFoundError: Dosya yoksa.
        ValueError: Dosya biçimi tanınmıyorsa.
    """
    import joblib

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {p}")
    payload = joblib.load(p)
    if not isinstance(payload, dict) or "booster_str" not in payload:
        raise ValueError(f"Tanınmayan model dosyası biçimi: {p}")
    booster = lgb.Booster(model_str=payload["booster_str"])
    return TrainedModel(
        booster=booster,
        feature_names=list(payload["feature_names"]),
        config=payload.get("config", {}),
        metadata=payload.get("metadata", {}),
    )


def check_feature_parity(model: TrainedModel, X: pd.DataFrame) -> None:  # noqa: N803
    """Canlı tarafta üretilen özelliklerin model şemasıyla aynı olduğunu doğrular.

    Canlı botun sinyal üretmeden ÖNCE çağırması gereken güvenlik kontrolü.

    Args:
        model: Yüklenmiş model.
        X: Canlı üretilen özellik matrisi.

    Raises:
        ValueError: Sütunlar eksik/fazla ise veya sıra farklıysa.
    """
    expected = list(model.feature_names)
    got = list(X.columns)
    if expected == got:
        return
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    raise ValueError(
        "FEATURE PARITY İHLALİ.\n"
        f"  Eksik sütunlar: {missing}\n"
        f"  Fazla sütunlar: {extra}\n"
        "  features.make_features aynı FeatureConfig ile çağrıldı mı?"
    )
