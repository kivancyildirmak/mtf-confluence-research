"""Paribu için ML tabanlı sinyal araştırma hattı (offline).

Modüller:

* :mod:`research.config` — tüm parametreler tek yerde.
* :mod:`research.data` — veri katmanı (CSV/parquet + Paribu API stub'ları).
* :mod:`research.features` — ORTAK özellik mühendisliği (feature parity).
* :mod:`research.labeling` — triple-barrier, meta-labeling, örnek ağırlıkları.
* :mod:`research.model` — LightGBM eğitimi, SHAP, kaydet/yükle.
* :mod:`research.validation` — purged CV, combinatorial purged CV, DSR.
* :mod:`research.backtest` — komisyon/slippage/gecikme dahil backtest.
* :mod:`research.sizing` — olasılık -> pozisyon boyutu.
* :mod:`research.run_research` — uçtan uca çalıştırıcı.

Kullanım:

    python -m research.run_research --synthetic
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "features",
    "labeling",
    "model",
    "validation",
    "backtest",
    "sizing",
]
