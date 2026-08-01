# mtf-confluence-research

TradingView Pine Script research

## Paribu ML sinyal araştırma hattı

`research/` klasöründe, Paribu için makine öğrenmesi tabanlı sinyal modeli
geliştirmeye yönelik **offline araştırma + backtest hattı** bulunur. Bu aşamada
canlı API bağlantısı yoktur; veri çekme ve emir gönderme fonksiyonları stub'dır.

```bash
pip install -r research/requirements.txt
python -m research.run_research --synthetic   # uçtan uca demo
python -m research.tests.test_pipeline        # sızıntı testleri dahil
```

Ayrıntılar: [research/README.md](research/README.md)
