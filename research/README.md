# Paribu ML Sinyal Araştırma Hattı (offline)

Makine öğrenmesi tabanlı bir kripto sinyal modelinin **araştırma ve backtest**
altyapısı. Bu aşamada **canlı API bağlantısı yoktur**: veri çekme ve emir
gönderme fonksiyonları imzaları ve dokümantasyonları hazır, gövdeleri
`NotImplementedError` olan stub'lardır. Odak, canlıya geçildiğinde sonuçların
tekrar edeceği **sağlam bir offline hat** kurmaktır.

---

## 1. Hızlı başlangıç

```bash
pip install -r research/requirements.txt

# Sentetik veriyle uçtan uca çalıştır
python -m research.run_research --synthetic

# Kendi verinizle
python -m research.run_research --data data/btc_tl_1m.parquet

# Testler (pytest gerekmez)
python -m research.tests.test_pipeline
```

Çıktılar `artifacts/` klasörüne yazılır: `model.pkl`, `report.json`,
`backtest.png`, `feature_importance.png`, `cpcv_distribution.png`, `trades.csv`
ve metrik tabloları.

---

## 2. Hattın akışı

```
data.py           Veri (yerel CSV/parquet | sentetik | TODO: Paribu API)
   |
features.py       ORTAK özellik üretimi  <-- canlı bot da BURAYI kullanacak
   |
labeling.py       CUSUM olayları -> triple-barrier -> meta-label
   |              + örnek benzersizliği ağırlıkları
   |
validation.py     Purged K-Fold (+embargo) -> OOS olasılıklar
   |              Combinatorial Purged CV -> performans DAĞILIMI
   |              Walk-forward -> nihai doğrulama
   |
model.py          LightGBM (ağırlıklı) + SHAP + kaydet/yükle
   |
sizing.py         olasılık -> bahis boyutu -> kapaklı Kelly -> vol hedefleme
   |
backtest.py       komisyon + slippage + GECİKME ile icra, metrikler
   |
validation.py     Deflated Sharpe Ratio (çoklu deneme düzeltmesi)
```

`run_research.py` bu sırayı uçtan uca çalıştırır ve özet tablo + grafik üretir.

---

## 3. En kritik kural: FEATURE PARITY

Backtest ile canlı botun **birebir aynı** hesabı yapması gerekir. Bunun için:

* Tüm özellikler **tek bir fonksiyondan** üretilir: `features.make_features()`.
  Canlı bot da yalnızca bunu çağırmalıdır — kopyalanmış/yeniden yazılmış bir
  gösterge kodu paritenin en yaygın bozulma sebebidir.
* Fonksiyon **deterministiktir**, global durum tutmaz, rastgelelik içermez.
* Emir defteri veya referans varlık verisi yoksa sütunlar **silinmez, NaN
  bırakılır**. Böylece özellik şeması her koşulda aynıdır (LightGBM NaN'ı
  doğal olarak işler).
* Sütun sırası alfabetik olarak sabitlenir.
* `model.save_model()` modeli, özellik listesini ve tüm konfigürasyonu **tek
  dosyada** saklar. Canlı taraf `model.check_feature_parity()` ile şemayı
  doğrulamadan sinyal üretmemelidir.

```python
from research.features import make_features
from research.model import load_model, check_feature_parity

m = load_model("artifacts/model.pkl")
X = make_features(son_barlar, orderbook=defter, reference=btc)
check_feature_parity(m, X)          # şema uyuşmazlığında ValueError
p = m.predict_proba(X.iloc[[-1]])   # yalnızca KAPANMIŞ son bar
```

---

## 4. Sızıntı (look-ahead / leakage) politikası

Hattın her katmanında uygulanan kurallar:

| Katman | Risk | Önlem |
|---|---|---|
| `data.py` | Kapanmamış son bar verinin içinde | `drop_unclosed_bar()` — canlıda zorunlu |
| `data.py` | Eksik barları sessizce doldurmak | `align_to_bars(fill_gaps=False)` varsayılan |
| `features.py` | `shift(-k)`, `center=True`, global ortalama/std | Yalnızca trailing `rolling`/`ewm`; global istatistik yok |
| `features.py` | Çapraz varlık aynı-bar getirisi | Ayrıca gecikmeli (`shift(lag)`) sürümler üretilir |
| `labeling.py` | Bariyer taramasının olay barını içermesi | Tarama `t+1`'den başlar |
| `labeling.py` | Ufku veri sonuna sığmayan olaylar | `NaT` işaretlenip **elenir** (kırpılmış etiket = yanlı etiket) |
| `labeling.py` | Meta-label için ML birincil model | Birincil model **deterministik kural** (`primary_side_rule`) |
| `validation.py` | Çakışan etiketler | **Purging**: `t1` üzerinden iki yönlü eleme |
| `validation.py` | Seri korelasyon | **Embargo**: test bloğu sonrası karantina |
| `backtest.py` | Sinyal barından işlem | **Gecikme**: `t + latency_bars` barının açılışı |
| `backtest.py` | Bariyerden tam fiyatla çıkış | Çıkış, temas barının SONRASINDAKİ açılıştan |
| `sizing.py` | Tüm örneklem volatilitesiyle ölçekleme | Yalnızca kayan pencere vol tahmini |

Bunların yorumda kalmaması için `research/tests/test_pipeline.py` içinde
**davranışsal testler** vardır. En önemlisi `test_features_are_causal`: seri
`k`. bardan kesildiğinde hesaplanan son satır, tüm seride hesaplanan aynı
satıra **birebir eşit** olmalıdır. Değilse o özellik geleceği görüyordur.

---

## 5. Etiketleme (triple-barrier + meta-labeling)

**Neden sabit ufuklu etiket değil?** "60 bar sonraki getirinin işareti" gerçek
bir işlemin nasıl kapandığını (zarar-kes / kâr-al) yansıtmaz.

1. **Olay örnekleme (CUSUM).** Her bar bir örnek değildir; kümülatif hareket
   eşiği aşınca olay üretilir. Etiket çakışmasını ciddi biçimde azaltır.
2. **Üç bariyer.** Üst = `pt_mult × hedef_vol`, alt = `sl_mult × hedef_vol`,
   dikey = `vertical_bars`. Etiket, **ilk değilen** bariyerdir. Aynı barda iki
   bariyer de değerse kötümser varsayım (zarar-kes) uygulanır.
3. **Meta-labeling.** Birincil model (deterministik EMA kuralı) **yönü** verir;
   LightGBM yalnızca "bu işleme gir / girme" ikili kararını öğrenir. Çıktı bir
   olasılıktır ve doğrudan pozisyon boyutuna çevrilir.
4. **Örnek ağırlıkları** (López de Prado): ortalama benzersizlik × getiri atfı
   × zaman sönümü. Ağırlıklar **yalnızca eğitimde** kullanılır; raporlanan
   metrikler ağırlıksızdır.

> **Bariyer genişliği maliyetten büyük olmalı.** Gidiş-dönüş maliyet ~%0.5 iken
> bariyeri %0.3'e kurmak, model ne kadar iyi olursa olsun matematiksel olarak
> kaybeden bir strateji üretir. Varsayılanlar (`pt_sl=(2.0, 2.0)`,
> `vertical_bars=240`) bu kısıt gözetilerek seçilmiştir.

---

## 6. Doğrulama

* **Purged K-Fold + embargo** — tüm örneklem için OOS olasılık üretir.
* **Combinatorial Purged CV** — `C(N,k)` bölünme, `C(N,k)·k/N` bağımsız
  backtest yolu. Tek skor yerine **dağılım** verir; her bölünmede maliyetli
  mini-backtest de çalıştırılır, yani dağılım AUC'de değil **parada** görülür.
* **Walk-forward** — dağıtımın en yakın taklidi: geçmişte eğit, hemen sonrasını
  test et, kaydır.
* **Deflated Sharpe Ratio** — gözlenen Sharpe'ı 0 ile değil, `n_trials` deneme
  sonrası **şans eseri beklenen maksimum Sharpe** ile karşılaştırır.

> `CVConfig.n_trials` dürüstçe doldurulmalıdır: denenen tüm özellik setleri,
> hiperparametreler, bariyerler, eşikler. Küçük beyan etmek DSR'yi şişirir ve
> testi anlamsızlaştırır.

---

## 7. Backtest varsayımları

Bilinçli olarak **kötümser**:

| Varsayım | Değer / davranış |
|---|---|
| Komisyon | `commission_rate` × 2 (giriş + çıkış), varsayılan %0.20 taker |
| Slippage | `slippage_bps`, fiyata yön **aleyhine** uygulanır |
| Gecikme | Sinyal bar `t` kapanışında; emir `t + latency_bars` **açılışında** |
| Çıkış | Bariyer teması bar kapanışında fark edilir, çıkış sonraki açılıştan |
| Eşzamanlılık | Varsayılan **tek pozisyon**; açıkken gelen sinyaller atlanır |
| Modellenmeyen | Kısmi dolum, piyasa etkisi, fonlama → sonuç bir **ÜST SINIRDIR** |

**Metrikler:** maliyet sonrası toplam getiri, CAGR, Sharpe, Sortino, maksimum
drawdown ve süresi, Calmar, işlem sayısı, isabet oranı, ortalama kazanç/kayıp,
kazanç/kayıp oranı, profit factor, toplam maliyet, maliyetin brüte oranı,
piyasada kalma oranı. Ayrıca **al-ve-tut ölçütü** ve **maliyet duyarlılık
analizi** (0×, 0.5×, 1×, 1.5×, 2× maliyet) raporlanır.

> **Accuracy'e güvenmeyin.** Dengesiz etiketlerde çoğunluk sınıfını söylemek
> yüksek accuracy verir. Karar, maliyet sonrası backtest ve DSR'dedir. Sharpe
> ise 1 dakikalık barlarda değil, `metric_freq` (varsayılan saatlik) frekansına
> indirgenerek hesaplanır — nakitte geçen barların sıfır getirisi oranı aksi
> halde yapay olarak şişirir.

---

## 8. Pozisyon boyutlandırma

`sizing.py` üç bileşeni birleştirir ve **en muhafazakârını** seçer:

1. Olasılık boyutu — `2·Φ((p−0.5)/√(p(1−p))) − 1`.
2. Kapaklı (fractional) Kelly — `f* = (p·b − (1−p))/b`, `kelly_fraction` ile
   kesirli, `kelly_cap` ile kapaklı. Tam Kelly **önerilmez**.
3. Volatilite hedefleme — `hedef_vol / tahmini_vol`, `max_leverage` ile sınırlı.

`method="combined"` (varsayılan): `min(olasılık, Kelly) × vol_ölçeği`.

---

## 9. Konfigürasyon

Tüm parametreler `config.py` içindedir; kodda sihirli sayı yoktur.
`DataConfig`, `FeatureConfig`, `LabelConfig`, `ModelConfig`, `CVConfig`,
`BacktestConfig`, `SizingConfig` → `ResearchConfig`.

Tekrarlanabilirlik: tek bir `RANDOM_SEED`, `set_global_seed()` ile Python/NumPy'a
ve `ModelConfig.params` üzerinden LightGBM'e dağıtılır (`deterministic: True`).
Config, eğitilen modelle birlikte diske yazılır.

---

## 10. Paribu API'sini bağlarken (TODO)

Doldurulacak üç fonksiyon `data.py` içindedir:

* `fetch_paribu_ohlcv(symbol, interval_minutes, start, end, limit)`
* `fetch_orderbook(symbol, depth, snapshot_interval_minutes, start, end)`
* `submit_order(symbol, side, quantity, order_type, price)`

Uyulması gerekenler:

1. Dönen veri `validate_ohlcv()` sözleşmesine uymalı (UTC `DatetimeIndex`,
   `open/high/low/close/volume`, artan ve tekilleştirilmiş).
2. **Kapanmamış son bar atılmalı** (`drop_unclosed_bar`).
3. Eksik barlar sessizce doldurulmamalı.
4. Emir defteri anlık görüntüsü **barın kapanış anına** hizalanmalı.
5. Canlı sinyal üretiminden önce `check_feature_parity()` çağrılmalı.
6. Isınma: elde `features.warmup_bars()` kadar geçmiş yoksa sinyal üretilmemeli.

### 10.1 Bağlantı durumu (2026-08-02) — HENÜZ BAĞLANMADI

Market-data uçlarını gerçek API'ye bağlama denemesi, **geliştirme ortamının
çıkış (egress) politikası** yüzünden tamamlanamadı. Doğrulanabilir olgular:

```
docs.paribu.com:443   CONNECT reddedildi (403)
www.paribu.com:443    CONNECT reddedildi (403)
api.paribu.com:443    CONNECT reddedildi (403)
v1.paribu.com:443     CONNECT reddedildi (403)
```

Bu Paribu'ya özel bir engel değil; ortam **katı bir allowlist** kullanıyor
(`example.com` ve `api.binance.com` de kapalı, yalnızca GitHub ve paket
kayıtları açık). Dolayısıyla resmî doküman okunamadı ve canlı istek atılamadı.

**Bilinçli karar:** Doğrulanmamış endpoint yolları koda gömülmedi. Tahmin
edilmiş bir yolu "dokümandan alındı" gibi göstermek, hattın tüm dürüstlük
kuralını çiğnerdi. Stub'lar `NotImplementedError` olarak duruyor; hangi bilginin
eksik olduğu docstring'lerinde açıkça yazılı.

**Engeli kaldırmanın iki yolu:**

1. `research/tools/probe_paribu.py` betiğini **kendi makinenizde** çalıştırın
   (bağımlılık yok, anahtar yok) ve `paribu_probe.json` çıktısını paylaşın:

   ```bash
   python3 research/tools/probe_paribu.py --dump-docs
   ```

   Betik aday uçları dener ve hangisinin gerçekten cevap verdiğini, yanıt
   yapısıyla birlikte raporlar. Listedeki adaylar **doğrulanmış yollar değil,
   test edilecek adaylardır**.

2. Ya da ortamın ağ politikasında `*.paribu.com` erişimine izin verin; o zaman
   dokümanı okuyup uçları doğrudan bağlayabilirim.

### 10.2 Tarihsel mum (OHLCV) verisi sorunu

Paribu'da tarihsel mum ucu olup olmadığı **doğrulanamadı** (doküman
okunamadı — yukarı bakın). Bu yüzden aşağıdaki iki seçenek, ucun var
olmaması ihtimaline karşı hazırdır. Not: emir defteri (`mk_*`) özellikleri için
tarihsel veri **hiçbir kaynakta yoktur**; onlar yalnızca ileriye dönük
toplamayla elde edilebilir.

**(a) Poll-forward toplayıcı — Paribu-native, ileriye dönük**

Public ucu düzenli aralıkla çekip bardan bar geçmiş biriktiren bir servis
(cron/systemd). Parquet'e ekler, tekilleştirir, UTC'ye çevirir.

* Artı: Paribu'nun **kendi** fiyatı, spread'i ve defteri. Mikroyapı özellikleri
  (`mk_spread_rel`, `mk_imbalance`, `mk_depth_*`) yalnızca böyle doldurulabilir.
* Eksi: **Sıfırdan başlar.** Hattın varsayılan ayarlarıyla anlamlı sayıda olay
  (~2.900) için kabaca **45 gün kesintisiz toplama** gerekir.
* Teknik uyarı: Barları *ticker* anlık görüntülerinden kurarsanız `high`/`low`
  **sistematik olarak dar** çıkar (yalnızca yoklama anlarını görürsünüz) ve bu
  Parkinson/Garman-Klass tahmincilerini bozar. **Trades (işlemler) ucundan
  toplayın**; gerçek işlemleri toplulaştırmak tam doğru OHLCV verir.

**(b) Üçüncü parti tarihsel BTC/TRY barları — anında derinlik**

Binance BTCTRY spot geçmişi public ve anahtarsızdır; toplu dökümü de vardır
(`data.binance.vision`). CryptoCompare/CoinAPI da BTC-TRY sunar.

* Artı: **Bugün** yıllarca geçmişe erişim; araştırmaya hemen başlanır.
* Eksi: Binance BTCTRY ≠ Paribu BTC_TL. Farklı likidite, farklı spread, farklı
  TL primi ve farklı mikroyapı. Fiyat dinamiği araştırması için makul bir vekil,
  **icra gerçekliği için değil**. `mk_*` sütunları NaN kalır.

**Öneri: ikisini birden, bu sırayla.**

1. **Bugün (a)'yı başlatın** — toplayıcı arka planda çalışsın, Paribu-native
   veri ve defter geçmişi birikmeye başlasın. Gecikilen her gün geri gelmez.
2. **Paralelde (b) ile araştırın** — model geliştirme, özellik seçimi ve
   hiperparametreler Binance BTCTRY üzerinde şekillensin.
3. **Canlıya geçmeden önce (a)'nın verisiyle doğrulayın.** Nihai karar
   (özellikle maliyet/slippage varsayımları ve `mk_*` özellikleri)
   Paribu-native veride verilmelidir; aksi halde backtest başka bir borsanın
   mikroyapısını ölçmüş olur.

> Yalnızca (b) ile canlıya geçmek, bu hattın kaçınmak için kurulduğu hatanın
> ta kendisidir: doğru görünen ama gerçek icra koşullarını ölçmeyen bir backtest.

### 10.3 Poll-forward toplayıcı (uygulandı)

Probe sonucu doğrulandı: Paribu'da **tek public market-data ucu** vardır.

| Uç | Durum |
|---|---|
| `GET https://www.paribu.com/ticker` | **çalışıyor** (anahtarsız) |
| candles / klines | **404 — yok** |
| orderbook | **404 — yok** |
| trades | **404 — yok** |

Ticker yanıtı parite başına: `last`, `lowestAsk`, `highestBid`, `low24hr`,
`high24hr`, `avg24hr`, `volume`, `change`, `percentChange`.

Tarihsel uç olmadığı için `research/tools/collect_paribu.py` geçmişi
**bugünden itibaren** biriktirir:

```bash
# Toplamayı başlat (Ctrl+C ile temiz kapanır)
python -m research.tools.collect_paribu collect

# Kesintisiz arka planda
nohup python -m research.tools.collect_paribu collect > collector.log 2>&1 &

# Ne kadar birikti?
python -m research.tools.collect_paribu status

# 1 dakikalık barlara çevir
python -m research.tools.collect_paribu resample --symbol BTC_TL
```

Ham tick'ler `research/data/ticks/YYYY-MM-DD.parquet` dosyalarına append-only
yazılır (geçici dosya + `os.replace` ile **atomik**, yani yazarken süreç ölse
bile dosya bozulmaz). Ayarlar `config.CollectorConfig` içindedir: pariteler,
yoklama aralığı (varsayılan **5 sn**), flush ve kalp atışı periyotları.

Sağlamlık: ağ hatasında süreç **düşmez**; üstel geri çekilme uygulanır
(2s → 4s → 8s … `backoff_max_seconds` sınırına kadar), başarılı okumada sıfırlanır.
`SIGINT`/`SIGTERM` alındığında tampon diske yazılır. Ani kapanmada en fazla
`flush_interval_seconds` (varsayılan 30 sn) kadarlık tick kaybedilir.

`data.fetch_paribu_ohlcv()` artık bu yerel depoyu okur (ağa gitmez) ve
standart OHLCV sözleşmesini uygular: UTC indeks, artan sıra, tekilleştirme,
**kapanmamış son bar atılır**.

#### Bu verinin üç yapısal sınırı

**1. `high`/`low` sistematik olarak DARDIR.** Barlar ticker anlık
görüntülerinden kurulur; 5 sn aralıkta dakikada ~12 örnek görülür. Bu örnekler
arasındaki gerçek ekstremler kaybolur. Sonuç: `vol_pk_*` (Parkinson) ve
`vol_gk_*` (Garman-Klass) tahmincileri **gerçek volatiliteyi düşük gösterir**;
`vol_rv_*` (kapanış-kapanış) bundan daha az etkilenir. Bar başına örnek
sayısını `tick_count` sütunundan izleyin.

**2. `volume` 24 SAATLİK KÜMÜLATİFTİR, anlık değil.** Dakikalık hacim ardışık
okumaların farkından türetilir ve bu iki senaryoda iki farklı şey demektir:

* *Günlük sıfırlanan sayaç:* negatif fark yalnızca gün dönümünde görülür ve
  fark = o aralığın gerçek hacmidir. **Kullanılabilir.**
* *Kayan 24s penceresi:* pencereye giren kadar çıkan işlem de vardır, yani
  fark = (yeni hacim) − (24 saat önce düşen hacim). Negatif farklar güne
  yayılır ve **fark artık aralık hacmi değildir (aşağı yanlı)**.

Hangisinin geçerli olduğunu varsaymıyoruz: `resample`, negatif farkların gün
dönümünde kümelenip kümelenmediğine bakarak modu **ölçer** ve raporlar
(`diagnose_volume_series`). Negatif farklar "ölçülemedi" sayılıp NaN yapılır —
uydurma değer üretilmez — ve `volume_gecerli_oran` sütunu her bar için
farkların ne kadarının kullanılabildiğini söyler. Bu oran 1.0'ın belirgin
altındaysa hacim özelliklerine (`volm_*`) temkinli yaklaşın.

**3. `avg24hr`/`change` gibi türetilmiş alanlar kullanılmaz** — hepsi 24 saatlik
pencereye bağlıdır ve dakikalık araştırma için bilgi taşımaz.

Buna karşılık **bir kazanç**: `lowestAsk`/`highestBid` sayesinde gerçek spread
ölçülür. `resample` çıktısındaki `spread_mean` ve `spread_rel_mean` sütunları,
emir defteri ucu olmamasına rağmen elimizdeki tek gerçek mikroyapı sinyalidir
ve maliyet/slippage varsayımlarını kalibre etmek için değerlidir.

#### Doğrulanmamış nokta: sayı biçimi

Paribu'nun sayıları hangi biçimde döndürdüğü (`3000000.5` mi, `"3.000.000,50"`
mi) canlı yanıt görülmeden doğrulanamadı. `_to_float` bu yüzden muhafazakâr
çalışır: **önce** standart `float()` denenir, **yalnızca o başarısız olursa** TR
biçimi (nokta=binlik, virgül=ondalık) denenir. Sıra kritiktir — baştan nokta
silmek `"3000.50"` değerini `300050` yapar, yani fiyatı 100 katına çıkaran
sessiz bir bozulma olurdu. İlk gerçek yanıt geldiğinde biçim netleşir; tek
belirsiz durum `"3.000"` gibi her iki biçimde de geçerli değerlerdir.

### 10.4 Binance prototip verisi — `fetch_binance.py`

Poll-forward toplayıcının hattın ölçeğine ulaşması ~45 gün sürer. O süreyi
beklemeden hattı **gerçek** (sentetik olmayan) kripto verisiyle sınamak için
`research/tools/fetch_binance.py` geçmiş 1 dakikalık bar indirir. Anahtarsızdır.

```bash
python -m research.tools.fetch_binance                  # ~60 gun, sembol olculerek secilir
python -m research.tools.fetch_binance --days 90
python -m research.tools.fetch_binance --symbol BTCUSDT --source rest
python -m research.tools.fetch_binance --probe-only     # sadece olc, indirme
```

> ## ⚠ BU VERİ PARİBU DEĞİLDİR
>
> Binance BTCTRY/BTCUSDT ile Paribu BTC_TL **farklı borsalardır**: farklı
> likidite, farklı spread, farklı TL primi, farklı mikroyapı, farklı icra.
> Burada elde edilen **hiçbir sonuç Paribu için geçerli sayılamaz** — ne Sharpe,
> ne isabet oranı, ne de maliyet varsayımları.
>
> Bu bir **PROTOTİPTİR**. Amacı yalnızca (1) hattın gerçek veriyle uçtan uca
> çalıştığını görmek ve (2) özellik/etiket parametrelerini kabaca kalibre
> etmektir. Sonuç ne kadar iyi görünürse görünsün, **canlıya geçmeden önce her
> şey Paribu'nun kendi verisiyle (bölüm 10.3, poll-forward) yeniden
> doğrulanmalıdır.**

**Kaynaklar (tercih sırasıyla).** Önce toplu döküm `data.binance.vision`
(tamamlanmış aylar için aylık zip, içinde bulunulan ay için günlük zip — tek
istekte bir ay, çok verimli); erişilemez veya boşsa REST
`api.binance.com/api/v3/klines` ile 1000'lik sayfalama. Toplu dökümde eksik
(404) dosya hata sayılmaz, atlanır ve raporlanır.

**Sembol seçimi ölçümle yapılır, varsayımla değil.** Önce `BTCTRY` denenir; son
2 gün indirilip **doluluk oranı** (gerçek bar / beklenen bar) hesaplanır. Oran
`MIN_COVERAGE` (%60) altındaysa parite "ince" sayılır ve otomatik olarak
`BTCUSDT`'ye düşülür. Hangisinin kullanıldığı hem loglanır hem de **dosya adına
yazılır** (`BTCTRY_1m_binance.parquet` / `BTCUSDT_1m_binance.parquet`), böylece
sonradan hangi veriyle çalışıldığı karışmaz.

İnce parite meselesi önemsiz değil: saatlerce işlem görmeyen bir seride model,
gerçekte olmayan boşlukları ve sahte volatilite rejimlerini öğrenir.

**Çıktı** `research/data/ohlcv/` altına parquet olarak yazılır ve hattın
sözleşmesine uyar: UTC `DatetimeIndex`, artan, tekrarsız, kapanmamış son bar
atılmış, sütunlar `open, high, low, close, volume`. Zaman damgası birimi
(ms/µs — Binance dökümlerde ikisini de kullanmıştır) **büyüklüğünden ölçülerek**
çıkarılır; varsaymak tarihleri 1000 kat kaydırıp veriyi sessizce çöpe çevirirdi.

Bu kaynakta emir defteri yoktur, dolayısıyla `mk_*` mikroyapı sütunları NaN
kalır — hat bunu zaten destekler (özellik şeması değişmez, LightGBM NaN'ı
doğal olarak işler). Gerçek spread verisi yalnızca Paribu toplayıcısından gelir.

Kullanım:

```bash
python -m research.tools.fetch_binance --days 60
python -m research.run_research --data research/data/ohlcv/BTCUSDT_1m_binance.parquet
```

---

## 11. Sentetik veri hakkında

`generate_synthetic_ohlcv()` stokastik volatilite (OU süreci), gün içi
mevsimsellik, Brownian köprüyle bar içi ekstremler ve volatiliteyle
korelasyonlu hacim üretir. İçinde **çok zayıf** bir momentum bileşeni vardır.

Sentetik veride hattın **kâr göstermemesi beklenen ve doğru sonuçtur**: orada
gerçek bir kenar (edge) yoktur ve bu altyapının asıl işi, olmayan bir kenarı
"var" göstermemektir. `run_research.py` sonunda basılan KARAR bloğu bunu açıkça
söyler. Gerçek veriyle çalışırken de aynı eşikler geçerlidir.

---

## 12. Bilinen sınırlar

* Kısmi dolum, piyasa etkisi ve emir defteri tüketimi modellenmez.
* Tek sembol, tek pozisyon; portföy düzeyi risk yönetimi yoktur.
* Hiperparametre araması yoktur (eklenirse `n_trials` güncellenmelidir).
* Model kalibrasyonu (Platt/isotonic) uygulanmaz; Kelly kalibre olasılık ister,
  bu yüzden `kelly_fraction` düşük tutulmuştur.
* Rejim değişimi tespiti yalnızca özellik seviyesindedir; ayrı bir rejim modeli
  yoktur.

---

## 11. Deney defteri ve bar boyutu deneyi

### 11.1 `n_trials` neden 120?

Deflated Sharpe, gözlenen Sharpe'ı **kaç deneme yapıldığına** göre cezalandırır.
Bu sayıyı küçük beyan etmek DSR'yi sahte biçimde yükseltir, yani tüm testi
anlamsızlaştırır. Defter:

**Uçtan uca değerlendirilen konfigürasyonlar (~8):** sentetik 1m (ilk ayar:
`pt_sl=1.5`, `vertical=60`, `cusum=1.0`, bar-bazlı Sharpe) · sentetik 1m
(yeniden ayarlanmış: `pt_sl=2.0`, `vertical=240`, `cusum=0.35`) · boyutlandırma
revizyonu (`kelly_fraction` 0.25→0.5, `target_vol` 0.20→0.30,
`min_position` 0.05→0.01) · metrik frekansı değişimi (bar → saatlik) · 1m
kontrol (600k bar) · 15m · 1h · (planlanan) Binance 1m.

**Belgelenen serbestlik dereceleri (~15):** bariyer çarpanları, dikey bariyer,
CUSUM eşiği, `prob_threshold`, `min_edge_over_cost`, boyutlandırma yöntemi,
Kelly kesri/kapağı, vol hedefi, maksimum kaldıraç, özellik grupları (7 grup),
birincil kural parametreleri, ağırlıklandırma seçenekleri, embargo oranı,
CPCV grup sayısı, metrik frekansı.

`8 × 15 = 120`. Bilinçli olarak **yukarı** yuvarlanmıştır: fazla beyan etmek
testi yalnızca sertleştirir, az beyan etmek sonucu sahte biçimde anlamlı
gösterir.

### 11.2 Bar boyutu deneyi (`--resample`)

**Hipotez:** 1 dakikalık barlarda yön neredeyse rastgeledir; daha uzun barlarda
mikroyapı gürültüsü ortalanır ve sinyal belirginleşir.

**Tasarım.** Aynı sentetik 1m seri (600.000 bar ≈ 416 gün) üç çözünürlükte
işlendi. **Tutma ufku her üçünde de 4 saat** olacak şekilde config'ten yeniden
türetildi (`scale_config_for_bars`), yani tahmin ufku sabit — değişen tek şey
bar çözünürlüğü. Tüm sızıntı korumaları, purging, embargo, maliyetli backtest,
CPCV ve DSR **aynen** korundu; hiçbir güvence gevşetilmedi.

```bash
python -m research.run_research --synthetic --bars 600000 --resample 15min
python -m research.run_research --synthetic --bars 600000 --resample 1h
```

**Sonuçlar (sentetik veri):**

| bar | olay | ufuk | CV AUC | CPCV AUC | CPCV Sharpe | poz. yol | işlem | isabet | getiri | Sharpe | maxDD | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1m (kontrol) | 27.232 | 240 bar | 0.510 | 0.509 | −3.56 | %0 | 1.182 | 0.397 | −21.9% | −6.21 | −22.0% | 0.00 |
| 15m | 7.581 | 16 bar | 0.502 | 0.503 | −2.87 | %0 | 892 | 0.379 | −21.1% | −5.76 | −21.2% | 0.00 |
| 1h | 2.795 | 4 bar | 0.519 | 0.520 | −1.44 | %7 | 675 | 0.373 | −11.8% | −3.09 | −12.7% | 0.00 |

**Yorum — hipotez DESTEKLENMEDİ.** AUC her üç çözünürlükte de 0.50–0.52
bandında kaldı; yani öngörülebilirlik artmadı. Sharpe'ın −6.2'den −3.1'e
"iyileşmesi" bir sinyal kazanımı değil, **daha az işlem yapıldığı için daha az
maliyet ödenmesidir**: işlem sayısı 1.182'den 675'e düşerken isabet oranı da
0.397'den 0.373'e geriledi. Üç çalıştırmanın da DSR'si ≈ 0'dır ve KARAR bloğu
beş kontrolün dördünde "KALDI" verir.

**Bu deneyin kanıtlayamayacağı şey.** Sentetik seride gerçek bir kenar yoktur
(zayıf bir AR(1) momentum bileşeni vardır ve toplulaştırma onu zaten ortalar).
Dolayısıyla bu sonuç, gerçek piyasada uzun barların işe yaramayacağını
**göstermez**; yalnızca hattın olmayan bir kenarı üretmediğini gösterir. Hipotez
ancak gerçek veriyle (Binance prototipi veya Paribu toplayıcısı) test edilebilir.

**Bilinen karışıklık (confound).** Özellik pencereleri BAR cinsinden sabit
tutuldu, duvar saatine göre ölçeklenmedi. Ölçeklenselerdi göstergeler bozulurdu
(1 saatlik barda MACD(12,26,9) → MACD(1,1,1), RSI(14) → RSI(1)). Bunun bedeli:
kaba barlarda özellikler duvar saati olarak daha geriye bakar, yani
karşılaştırma saf "çözünürlük" değil "çözünürlük + geriye bakış" birleşimidir.

---

## 12. Grid (ızgara) stratejisi — `grid_backtest.py`

Mekanik grid stratejisinin maliyetli ve **dürüst** backtest'i. ML hattından
bağımsızdır (tahmin yok, kural var) ama aynı maliyet disiplinini kullanır.

```bash
python -m research.grid_backtest                      # varsayilan: BTCTRY 1m -> 1h
python -m research.grid_backtest --levels 30 --capital 50000
python -m research.grid_backtest --synthetic --plot artifacts/grid.png
```

Veri `research/data/ohlcv/BTCTRY_1m_binance.parquet`'ten okunur ve 1 saatlik
bara indirilir (1 dakikalık gridde işlem/maliyet patlar). Dosya yoksa araç ne
yapılacağını söyler; `--synthetic` yalnızca motoru göstermek içindir.

### Neden bu modül "toplam kâr" raporlamıyor

Grid backtest'leri sektörde sistematik olarak yanıltıcıdır: genellikle sadece
**gerçekleşmiş kâr** (tamamlanan al-sat çiftleri) raporlanır ve bu sayı
neredeyse her zaman pozitiftir. Fiyat düştükçe grid almaya devam eder, her
yükselişte kâr kilitler. Gizlenen şey dönem sonunda **elde kalan coin**'dir.
Bu yüzden üç kalem AYRI raporlanır:

```
Gerceklesmis kar  :       2,912.66 TL  (+2.91%)
Gerceklesmemis K/Z:      -7,260.99 TL  (-7.26%)
TOPLAM K/Z        :      -4,348.33 TL  (-4.35%)
  DIKKAT: gerceklesmis kar POZITIF ama toplam NEGATIF.
  Yalnizca 'gerceklesmis kar' raporlansaydi strateji karli gorunecekti.
```

Ayrıca raporlanır: **aralık kırıldı mı** (fiyat alt sınırın altına indiyse
strateji fiilen çökmüştür), aralık dışı geçen sürenin oranı, maksimum drawdown,
elde kalan envanterin değeri/maliyeti, toplam maliyet, nakit yetmediği için
atlanan alışlar ve **al-ve-tut karşılaştırması**.

**Kademe adımı / maliyet oranı** başta basılır. Adım gidiş-dönüş maliyetten
küçükse strateji matematiksel olarak kaybeder — model ya da rejim fark etmez.

### İcra varsayımları (kötümser)

1. Her işlemde komisyon + slippage (varsayılan gidiş-dönüş ~%0.5, ML
   backtest'iyle aynı).
2. Aynı barda alınan lot aynı barda **satılamaz** — bar içi sıralama
   bilinmediği için sahte scalping üretilmez.
3. Piyasa üstü limit alım verilmez: bir kademede alım ancak fiyat oraya
   **düştüğünde** tetiklenir.
4. Nakit yetmezse alım **atlanır** ve sayılır (bedava kaldıraç yok).
5. Bir lotun satış hedefi **alım anında sabitlenir**; grid kaysa bile açık
   pozisyonun çıkış fiyatı geriye dönük değişmez.

Statik gridde sermaye kademelere önceden bölündüğü için (`sermaye / n_levels`)
alımların toplamı tanım gereği sermayeyi aşamaz; nakit tükenmesi ancak
maliyetlerle veya trailing modda yeniden alımla ortaya çıkar.

### Modlar

* **static** — aralık sabit. Fiyat aralıktan çıkarsa strateji durur.
* **trailing** — fiyat üst sınırı aşınca grid bir kademe yukarı kayar (en alt
  kademe boşsa). Yükselişte daha iyi, düşüşte aynı riski taşır.

İkisi de ayrı çalıştırılıp karşılaştırma tablosunda yan yana raporlanır.

### Rejim analizi (dönem seçme yanlılığına karşı)

Tek bir 60 günlük toplam sayı, hangi rejimin baskın olduğunu gizler. Bu yüzden
dönem haftalık pencerelere bölünür; her pencere için getiri, drawdown, fiyatın
net değişimi ve **rejim etiketi** (`yukselis` / `dusus` / `yatay`) raporlanır.
Beklenen ve gözlenen davranış: grid yatay piyasada kazanır, düşüşte kaybeder.

---

## 13. Ölçüm araçları (işlem yapmaz, anahtarsız)

İkisi de yalnızca **public fiyat okur**. Emir göndermez, anahtar/HMAC
kullanmaz, `submit_order`'a dokunmaz. Amaçları "fırsat var mı?" sorusunu
**ölçmektir** — kâr vaat etmek değil.

### 13.1 Üçgen arbitraj — `tools/triangular_paribu.py`

```bash
python -m research.tools.triangular_paribu
python -m research.tools.triangular_paribu --list-only
python -m research.tools.triangular_paribu --watch 300 --interval 5
python -m research.tools.triangular_paribu --start TL --legs
```

**Önce parite evreni raporlanır.** Üçgen için coin-coin paritesi (örn.
`BTC_USDT`) şarttır; borsada her şey TL'ye karşı işlem görüyorsa
`TL -> X -> Y -> TL` döngüsünün orta bacağı yoktur ve üçgen **yapısal olarak
imkânsızdır**. Araç bunu açıkça yazar ve ölçüm yapmadan durur.

**Gerçekçi fiyatlama.** Orta fiyat (mid) kullanmak bu işin en yaygın hatasıdır
ve olmayan fırsatları var gösterir. Her bacak gerçekte dolacağı taraftan
fiyatlanır: alıyorsan `lowestAsk` ödersin, satıyorsan `highestBid` alırsın.
Spread üç bacakta üç kez aleyhine çalışır.

**Komisyon zorunlu.** 3 işlem × `%0.2` = `%0.6`. Örnek çıktı (gerçekçi
spread'lerle, tutarlı fiyatlar):

```
  dongu                            brut %      net %  kar
  TL -> USDT -> BTC -> TL         -0.1201    -0.7182  hayir
  TL -> ETH -> USDT -> TL         -0.1279    -0.7259  hayir

  >>> NET POZITIF DONGU YOK. Su anda uygulanabilir firsat BULUNMUYOR.
      Brut getiri (-0.1201%) komisyon HARIC bile negatif:
      fiyatlar tutarli, aradaki fark tamamen spread'ten geliyor.
```

Net pozitif döngü **yoksa bunu açıkça yazar**; "fırsat var" gibi göstermez.
Fırsat çıktığında da uyarır: ölçüm anlıktır ve **defter derinliği kontrol
edilmez** — ilan edilen ask/bid yalnızca en üst seviyedir.

**`--watch`**: fırsatlar anlıktır, tek ölçüm yanıltır. Bu mod süre boyunca
örnekler ve kaç ölçümde net-pozitif fırsat çıktığını, döngü bazında ortalama/en
iyi net getiriyi sayar.

### 13.2 TL primi — `tools/cross_exchange.py`

```bash
python -m research.tools.cross_exchange
python -m research.tools.cross_exchange --watch 600 --interval 10
```

`TL primi = (Paribu_BTC_TL / global_BTC_TL - 1) × 100`, burada global fiyat
Binance'ten gelir: önce doğrudan `BTCTRY` denenir, yoksa `BTCUSDT × USDTTRY`.

> **Bu bir gözlem aracıdır, işlem stratejisi değildir.** Ölçülen fark **tek
> borsayla yakalanamaz**: iki borsada da aynı anda bakiye tutmak, transfer
> süresi + çekim ücretlerini göze almak, transfer sırasında primin kapanma
> riskini üstlenmek ve her iki tarafta komisyon + spread ödemek gerekir. Prim,
> TL'nin konvertibilite maliyetini ve yerel talebi yansıtır; **kalıcı olması
> normaldir ve kalıcı olması onu ücretsiz para yapmaz.** Bu uyarı aracın
> çıktısının en başında da basılır.

`--watch` primin ortalamasını, standart sapmasını, min/maks ve medyanını verir.
Std sapma ortalamanın çeyreğinden küçükse araç "prim dalgalanmıyor, kalıcı bir
seviye" yorumunu yapar — yani arbitraj fırsatı değil, yapısal fiyat farkı.

### 13.3 Ağsız test edilebilirlik

Her iki aracın da hesap çekirdeği **saf fonksiyonlardır**; ağ yalnızca ince bir
katmandır ve testlerde enjekte edilir (`getter` / `PriceSources`). Böylece
üçgen matematiği, ask/bid yönü, komisyon düşümü, prim formülü ve `--watch`
sayaçları internet olmadan doğrulanır. `--watch` döngüleri `sleeper`/`clock`
enjeksiyonuyla anında test edilir.
