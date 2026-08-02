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

#### Doğrulanmamış tek nokta: sayı biçimi

Paribu'nun sayıları hangi biçimde döndürdüğü (`3000000.5` mi, `"3.000.000,50"`
mi) canlı yanıt görülmeden doğrulanamadı. `_to_float` bu yüzden muhafazakâr
çalışır: **önce** standart `float()` denenir, **yalnızca o başarısız olursa** TR
biçimi (nokta=binlik, virgül=ondalık) denenir. Sıra kritiktir — baştan nokta
silmek `"3000.50"` değerini `300050` yapar, yani fiyatı 100 katına çıkaran
sessiz bir bozulma olurdu. İlk gerçek yanıt geldiğinde biçim netleşir; tek
belirsiz durum `"3.000"` gibi her iki biçimde de geçerli değerlerdir.

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
