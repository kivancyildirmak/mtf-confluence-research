# MTF Confluence System — Mimari Dokümanı

> Pine Script v6 · Yalnızca LONG · Repaint yok · Yalnızca kapanmış mumlar
>
> Bu doküman **Aşama 1** çıktısıdır: kod yok, yalnızca tasarım. Her modül ilerleyen
> aşamalarda tek tek geliştirilecektir. Hiçbir ücretli indikatör kopyalanmaz veya
> reverse-engineer edilmez; aynı problemleri çözen **özgün matematiksel yöntemler**
> tasarlanır.

---

## 0. Tasarım İlkeleri ve Kısıtlar

| İlke | Uygulama |
|------|----------|
| **Yalnızca LONG** | Tüm sinyal mantığı yükseliş yapısına odaklanır; short tarafı üretilmez. |
| **Repaint yok** | Hiçbir değer geçmişe dönük değişmez. Pivotlar yalnızca **onay** sonrası kesinleşir. |
| **Yalnızca kapanmış mumlar** | Tüm hesaplamalar `barstate.isconfirmed` mantığıyla, geçmiş (kapanmış) bar verisiyle yapılır. Anlık bar (`[0]`, canlı) karar üretmez. |
| **MTF güvenliği** | `request.security` çağrıları `lookahead=barmerge.lookahead_off` + `[1]` offset ile kullanılır. |
| **Modülerlik** | Her modül, saf (yan etkisiz) fonksiyonlar kümesidir. Global durum minimumda tutulur. |
| **Yasaklı göstergeler** | EMA, SMA, RSI, MACD, Supertrend, OBV, CMF, Mansfield ve hazır trend göstergeleri **kullanılmaz.** Trend/momentum, piyasa yapısı (market structure) ve pivot geometrisi üzerinden türetilir. |

### Yasaklı göstergelere özgün alternatifler

| Klasik ihtiyaç | Yasaklı çözüm | Bu sistemdeki özgün çözüm |
|----------------|---------------|---------------------------|
| Trend yönü | EMA/SMA/Supertrend | **Swing yapısı**: onaylı pivotlardan HH/HL/LH/LL dizisi (market structure state machine) |
| Momentum | RSI/MACD | **Pivot ivmesi**: ardışık legler arasındaki fiyat/zaman eğiminin değişim oranı |
| Hacim onayı | OBV/CMF | **Range-normalized effort**: (kapanış konumu × bar aralığı) ile bar-içi baskı ölçümü, ATR ile normalize |
| Volatilite/risk | — | **ATR** (izinli; bir gösterge değil, volatilite ölçüsüdür) |

---

## 1. Yüksek Seviye Mimari

```
                         ┌────────────────────────────────────────┐
                         │              GİRDİ KATMANI               │
                         │  OHLC (kapanmış bar) · ATR · TF context  │
                         └───────────────────┬────────────────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
        ▼                                    ▼                                    ▼
┌───────────────┐                   ┌───────────────┐                    ┌────────────────┐
│  M1: PIVOT     │                   │  M1: PIVOT    │                    │  M6: HTF        │
│  ENGINE (MAIN) │                   │  ENGINE (FAST)│                    │  CONTEXT/VETO   │
│  büyük eşik    │                   │  küçük eşik   │                    │  (üst TF pivot) │
└───────┬───────┘                    └──────┬───────┘                    └───────┬────────┘
        │  onaylı pivot akışı                │                                    │
        └──────────────┬─────────────────────┘                                    │
                       ▼                                                          │
              ┌──────────────────┐                                               │
              │ M2: MARKET        │                                               │
              │ STRUCTURE         │  HH/HL/LH/LL, trend state                     │
              └────────┬─────────┘                                               │
                       ▼                                                          │
              ┌──────────────────┐                                               │
              │ M3: XABCD BUILDER │  son 5 alternatif pivot → X,A,B,C,D           │
              └────────┬─────────┘                                               │
                       ▼                                                          │
              ┌──────────────────┐                                               │
              │ M4: HARMONIC      │  Fib oran eşleştirme + hata skoru             │
              │ PATTERN MATCHER   │                                               │
              └────────┬─────────┘                                               │
                       ▼                                                          │
              ┌──────────────────┐         ┌──────────────────┐                  │
              │ M5: MTF PIVOT     │◄────────│ Alt/üst TF pivot  │                  │
              │ CONFLUENCE        │         │ seviyeleri        │                  │
              └────────┬─────────┘         └──────────────────┘                  │
                       ▼                                                          │
              ┌──────────────────────────────────────────┐                       │
              │ M7: WEIGHTED CONFLUENCE SCORE             │◄──────────────────────┘
              │ ağırlıklı normalize skor (0..100)         │   (veto girişi)
              └────────┬─────────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │ M8: RISK ENGINE   │  ATR stop/target, RR, pozisyon boyutu
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ M9: SIGNAL &      │  giriş kararı + strateji backtest
              │ STRATEGY LAYER    │
              └──────────────────┘
```

---

## 2. Veri Akışı (Data Flow)

Her bar **kapandığında** aşağıdaki tek yönlü akış çalışır. Hiçbir adım geleceğe
bakmaz; her adım yalnızca *onaylanmış* geçmiş veriyi tüketir.

1. **Girdi normalizasyonu** — `high/low/close/open` + `ATR(len)` hesaplanır.
2. **Pivot motorları** (Main & Fast) yeni bir pivotu *onaylar* veya onaylamaz.
   Onay olayı gerçekleştiğinde pivot, halka tamponuna (ring buffer) yazılır.
3. **Market Structure** güncellenir: yeni onaylı pivot HH/HL/LH/LL olarak sınıflanır,
   trend durumu (state) güncellenir.
4. **XABCD Builder** son 5 alternatif (tepe/dip sıralı) pivotu okuyup aday yapı kurar.
5. **Harmonic Matcher** aday yapının leg oranlarını hesaplayıp desen şablonlarıyla
   karşılaştırır, bir *desen skoru* + *hata metriği* üretir.
6. **MTF Confluence** mevcut fiyatın alt/ana/üst TF pivot kümelerine yakınlığını
   ölçüp bir *confluence yoğunluğu* üretir.
7. **HTF Veto** üst TF yapısını değerlendirir; yükseliş değilse LONG bloklanır.
8. **Weighted Score** tüm alt skorları ağırlıklı toplayıp 0–100 aralığında tek
   skora indirger.
9. **Risk Engine** ATR ve yapı seviyelerinden stop/target/RR/pozisyon boyutu üretir.
10. **Signal & Strategy** eşik + veto + risk filtresi geçerse LONG üretir; strateji
    modunda `strategy.entry/exit` çağrılır.

**Repaint garantisi:** 2. adımdaki "onay" olayı geçmişte sabittir çünkü onay
koşulu (deviation + bar sayısı) yalnızca kapanmış barlarla değerlendirilir. Bir
pivot onaylandıktan sonra indeksi ve fiyatı bir daha değişmez.

---

## 3. Modül Detayları

### M1 — Pivot Engine (Onaylanmış ZigZag) · İki Bağımsız Motor

**Amaç:** Gürültüye dayanıklı, *repaint yapmayan* swing tepe/dip serisi üretmek.

**Özgün yaklaşım — "Confirmed Deviation ZigZag":**
Klasik `ta.pivothigh/pivotlow` yalnızca simetrik bar penceresi kullanır ve geç
kalır; klasik ZigZag ise canlı barda repaint eder. Burada iki koşullu bir
**durum makinesi** kullanılır:

- **Durumlar:** `SEEKING_HIGH`, `SEEKING_LOW`.
- Bir aday tepe (`SEEKING_HIGH` modunda çalışan max), fiyat bu tepeden
  **θ = k · ATR** kadar *aşağı* geri çekildiğinde **onaylanır**. Simetrik olarak dip.
- Ek gate: onay için adaydan bu yana **en az `minBars`** kapanmış bar geçmiş olmalı
  (mikro gürültü filtresi).
- Onay anı geçmişteki aday indeksine sabitlenir → **repaint yok** (indeks/fiyat
  bir daha oynatılmaz; yalnızca "onaylandı" bayrağı ileri bir barda set edilir).

**İki motor, aynı algoritma, farklı parametre:**

| Parametre | Main Engine | Fast Engine |
|-----------|-------------|-------------|
| Deviation `k` (× ATR) | büyük (örn. 3.0) | küçük (örn. 1.0) |
| `minBars` | büyük (örn. 8) | küçük (örn. 3) |
| Rolü | yapısal iskelet, HTF context | tetik zamanlaması, ince giriş |

Motorlar **tamamen bağımsızdır** (ayrı durum + ayrı ring buffer). Fonksiyon
imzası ortak olur; parametreyle örneklenir.

**Çıktı (pivot kaydı):**
```
Pivot { price, barIndex, time, kind(HIGH|LOW), confirmedAtIndex }
```
Ring buffer olarak son N pivot tutulur (örn. 16). Pine v6'da bu, `array<>` +
`udt`/`type` ile modellenir.

**Neden repaint etmez:** onay kriteri geleceğe değil, *geçmiş* deviation'a bakar;
`confirmedAtIndex ≥ barIndex + gecikme` olur ve bu gecikme kabul edilir (geç ama
kesin). Sinyal yalnızca `confirmedAtIndex` barında doğar.

---

### M2 — Market Structure (Trend State — EMA/Supertrend'siz)

**Amaç:** Trend yönünü *pivot geometrisinden* türetmek (yasaklı trend göstergesi yok).

**Yöntem:** Main motorun son onaylı tepe/dip serisinden:
- Tepe dizisi artıyorsa → **HH**, azalıyorsa **LH**.
- Dip dizisi artıyorsa → **HL**, azalıyorsa **LL**.
- **Trend state:**
  - `UPTREND` ⇔ son HH ve son HL (higher-high + higher-low).
  - `DOWNTREND` ⇔ LH + LL.
  - `RANGE` ⇔ karışık.
- **Yapı kırılımı (BoS/CHoCH benzeri, özgün eşiklerle):** fiyatın son onaylı tepe
  üstünde kapanışı = yükseliş yapı teyidi. (Yalnızca kapanışla, intrabar wick ile değil.)

**Çıktı:** `structureState ∈ {UP, DOWN, RANGE}`, `bosConfirmed(bool)`,
`swingStrength` (leg büyüklüklerinin ATR-normalize ölçüsü).

Bu modül LONG-only mantığın **birincil kapısıdır**: yalnızca UP/BoS bağlamında
harmonic girişlere izin verilir.

---

### M3 — XABCD Builder

**Amaç:** Onaylı pivot akışından X-A-B-C-D 5-nokta yapısını kurmak.

**Yöntem:**
- Pivotların tepe/dip **alternasyonu** garanti edilir (motor zaten alternatif üretir).
- Son 5 alternatif pivot → `X, A, B, C, D` (D en yeni onaylı pivot).
- LONG bağlamı için hedeflenen şablon: **D bir dip** olacak şekilde (bullish
  tamamlanma), yani `X(dip)-A(tepe)-B(dip)-C(tepe)-D(dip)` yönelimi.
- Legler: `XA, AB, BC, CD` — her biri fiyat farkı + bar farkı (Δprice, Δbars).

**Geçerlilik ön-filtreleri (harmonic'ten önce ucuz eleme):**
- Monotonluk: B, X'i aşmamalı; D, uygun retracement bölgesinde olmalı.
- Minimum leg boyu: her leg ≥ `m · ATR` (dejenere üçgenleri ele).

**Çıktı:** `XABCD { X,A,B,C,D pivotları, leg oranları }` veya `na` (geçersiz).

---

### M4 — Harmonic Pattern Matcher

**Amaç:** XABCD leg oranlarını harmonik şablonlara eşleştirmek — **kopya değil,
oran-hata metriğiyle özgün skorlama**.

**Oranlar (Fibonacci geometrisi — kamuya açık matematik):**
- `AB/XA`, `BC/AB`, `CD/BC`, `AD/XA` retracement/extension oranları.

**Şablon tablosu (bullish tamamlanmalar):** her desen için hedef oran aralığı
`[lo, hi]`. Örnek desen isimleri: Gartley, Bat, Butterfly, Crab, Cypher, Shark.
(Şablon oranları herkese açık Fibonacci sabitleridir; bir ücretli göstergeden
alınmaz.)

**Özgün skorlama — "Ratio Distance Score":**
Her oran `r_i` için hedef merkez `c_i` ve tolerans `t_i`:
```
err_i   = |r_i - c_i| / t_i          # normalize sapma (0 = tam isabet)
match_i = max(0, 1 - err_i)          # 0..1
patternScore = 100 · (Σ w_i·match_i) / (Σ w_i)   # ağırlıklı ortalama
```
- Herhangi bir oran toleransı aşarsa (`err_i > 1`) o desen için `match_i=0` → sert eleme.
- En yüksek `patternScore` veren desen seçilir; skor `harmonicScore` olarak dışarı verilir.
- **PRZ (Potential Reversal Zone):** D civarında Fib projeksiyonlarının
  kümelendiği fiyat bandı → risk motoruna girdi.

**Repaint yok:** oranlar yalnızca onaylı pivot fiyatlarından hesaplanır.

---

### M5 — MTF Pivot Confluence

**Amaç:** Mevcut fiyatın **birden çok zaman diliminden** gelen pivot seviyelerine
yakınlığını ölçmek. Farklı TF'lerdeki pivotların üst üste gelmesi = güçlü seviye.

**Zaman dilimleri:** `LTF (mevcut)`, `MTF (ara)`, `HTF (üst)`. Her biri için M1
motoru `request.security` ile çağrılır:
```
lookahead = barmerge.lookahead_off, ifade [1] offset ile → repaint yok
```

**Özgün yoğunluk metriği — "ATR-normalized cluster density":**
- Tüm TF'lerden son N pivot seviyesi tek havuza toplanır.
- Mevcut fiyat `p` için her seviye `L_j`'ye ağırlıklı yakınlık:
```
prox_j = exp( -( (p - L_j)^2 ) / (2·(σ·ATR)^2 ) )   # Gaussian çekirdek
confluenceDensity = Σ tf_weight_j · prox_j
```
- `tf_weight_j`: üst TF pivotlarına daha yüksek ağırlık.
- Sonuç 0..1'e normalize edilip `mtfScore` olur.

Böylece "üç farklı TF'de aynı bölgede pivot var" durumu yüksek skor üretir.

---

### M6 — HTF Context & Veto

**Amaç:** Üst zaman dilimi yapısı LONG'a karşıysa girişi **veto** etmek.

**Yöntem:**
- HTF üzerinde M2 (market structure) çalıştırılır (yine `lookahead_off`, `[1]`).
- **Veto koşulları (herhangi biri doğruysa LONG bloklanır):**
  1. HTF `structureState == DOWN`.
  2. Fiyat HTF son onaylı tepe ile dip arasında ama momentum aşağı (pivot ivmesi < 0).
  3. HTF'de yakın zamanda LL teyidi.
- Veto çıktısı boolean `htfVetoLong`. Ayrıca yumuşak bir `htfBias ∈ [-1..+1]`
  üretilir (skor motoruna ağırlık olarak girer).

Veto **sert filtredir**: skor ne olursa olsun `htfVetoLong == true` ise sinyal yok.

---

### M7 — Weighted Confluence Score

**Amaç:** Tüm alt sinyalleri tek, yorumlanabilir 0–100 skora indirmek.

**Girdiler (her biri 0..1'e normalize):**
| Bileşen | Kaynak | Örnek ağırlık `w` |
|---------|--------|-------------------|
| `structure` | M2 (UP + BoS gücü) | 0.25 |
| `harmonic` | M4 (patternScore/100) | 0.25 |
| `mtf` | M5 (confluenceDensity) | 0.20 |
| `htfBias` | M6 (pozitife map'lenmiş) | 0.15 |
| `effort` | range-normalized baskı (OBV/CMF alternatifi) | 0.10 |
| `momentum` | pivot ivmesi (RSI alternatifi) | 0.05 |

```
score = 100 · ( Σ w_i · s_i ) / ( Σ w_i )          # s_i ∈ [0,1]
```
- Ağırlıklar `input` ile ayarlanabilir → araştırma/optimizasyon.
- **Sert kapılar** skordan ayrı: `htfVetoLong`, `structureState==UP`,
  `harmonic geçerli`. Bunlar sağlanmadan skor değerlendirilmez.
- Karar eşiği: `score ≥ entryThreshold` (örn. 65).

---

### M8 — Risk Engine (ATR tabanlı)

**Amaç:** Giriş, stop, hedef ve pozisyon boyutunu volatiliteye ve yapıya göre üretmek.

**Kurallar:**
- **Entry:** onay barının kapanışı (yalnızca kapanmış mum).
- **Stop:** `min( D_pivot_low, entry − s·ATR )` — yapı dibinin ATR tamponu altında.
- **Target(lar):**
  - `TP1 = entry + R·(entry − stop)` (sabit RR, örn. R=1.5),
  - `TP2 = Fib projeksiyonu` (CD legi uzantısı / önceki HH bölgesi).
- **Pozisyon boyutu:** `qty = (equity · riskPct) / (entry − stop)` (risk-parity).
- **Trailing (opsiyonel):** yeni onaylı HL altına ATR-tamponlu stop yükseltme.
- Tüm seviyeler onay barında sabittir → repaint yok.

**Çıktı:** `RiskPlan { entry, stop, tp1, tp2, qty, rr }`.

---

### M9 — Signal & Strategy Layer (Backtest)

**Amaç:** Kararı üretmek ve `strategy` çerçevesinde test etmek.

**Giriş kararı (LONG):**
```
LONG ⇔  structureState==UP
     ∧ harmonicValid
     ∧ !htfVetoLong
     ∧ score ≥ entryThreshold
     ∧ riskPlan.rr ≥ minRR
```
- Sinyal yalnızca **onay barında** doğar; giriş bir sonraki barın açılışında
  (`strategy.entry`) veya kapanış bazlı, ama her hâlükârda gelecek verisi yok.

**Backtest mimarisi:**
- `strategy(...)` başlığı; `calc_on_every_tick=false`, `process_orders_on_close`
  ayarları repaint-safe kullanılır.
- Çıkışlar: `strategy.exit` ile stop + iki hedef (kısmi kapama).
- **Performans metrikleri** araştırma için tablo/label ile raporlanır: net profit,
  profit factor, win rate, maxDD, ortalama RR, sinyal sayısı.
- İki mod tek dosyada: `indicator` (görsel) vs `strategy` (test) — `input`
  ile derleme-zamanı seçimi yerine ayrı dosya önerilir (M-strategy / M-indicator).

---

## 4. Modül–Fonksiyon Haritası (uygulama iskeleti)

Her modül saf fonksiyonlar + gerekli minimal `var` durum. İsimler taslaktır.

| Modül | Anahtar fonksiyonlar (taslak) |
|-------|-------------------------------|
| M1 Pivot | `f_pivot_engine_step(state, k, minBars) → (state, newPivot?)`, `f_pivot_push(buf, p)` |
| M2 Structure | `f_classify_pivot(buf) → kind`, `f_structure_state(buf) → state`, `f_bos_confirmed(buf, close)` |
| M3 XABCD | `f_build_xabcd(buf) → xabcd?`, `f_leg_ratios(xabcd) → ratios` |
| M4 Harmonic | `f_match_patterns(ratios) → (name, score, prz)`, `f_ratio_distance(r,c,t)` |
| M5 MTF | `f_htf_pivots(tf) → levels`, `f_cluster_density(price, levels, atr) → 0..1` |
| M6 Veto | `f_htf_structure(tf) → state`, `f_htf_veto(state, mom) → bool`, `f_htf_bias(...) → -1..1` |
| M7 Score | `f_normalize(x)`, `f_weighted_score(components, weights) → 0..100` |
| M8 Risk | `f_atr_stop(...)`, `f_targets(...)`, `f_position_size(...)` |
| M9 Signal | `f_long_decision(...)`, strateji giriş/çıkış blokları |

---

## 5. Repaint & Güvenlik Kontrol Listesi

- [ ] Tüm pivot onayları geçmiş deviation ile → onay sonrası pivot sabit.
- [ ] `request.security` çağrılarında `lookahead_off` + `[1]` offset.
- [ ] Sinyal yalnızca `barstate.isconfirmed` / kapanmış barda doğar.
- [ ] `strategy` tarafında `process_orders_on_close` uyumu.
- [ ] Hiçbir `[0]` (canlı bar) değeri karar mantığına girmez.
- [ ] Higher-TF değerleri "gerçekleşmiş" (confirmed) HTF barından gelir.

---

## 6. Geliştirme Sırası (Aşama 2+)

1. **M1** Pivot Engine (Main+Fast) — çekirdek; her şey buna bağlı.
2. **M2** Market Structure — LONG kapısı.
3. **M3 + M4** XABCD & Harmonic — sinyal üreticisi.
4. **M5 + M6** MTF Confluence & HTF Veto — bağlam katmanı.
5. **M7** Weighted Score — birleştirme.
6. **M8** Risk Engine.
7. **M9** Strategy/Backtest + raporlama.

Her modül izole test edilebilir olacak (görsel doğrulama için indikatör
modunda plot/label ile).

---

## 7. Açık Tasarım Soruları (Aşama 2 öncesi karar)

1. Hedef enstrüman/zaman dilimi seti (örn. kripto 15m + 1h + 4h)?
2. Harmonik desen kapsamı: tam set mi (Gartley…Shark) yoksa çekirdek 3 desen mi?
3. Çıkış politikası: sabit RR mi, Fib hedefleri mi, trailing mi (yoksa hibrit)?
4. Tek dosya (indicator+strategy toggle) mı, ayrı iki dosya mı?
