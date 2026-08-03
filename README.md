# mtf-confluence-research

TradingView Pine Script research.

## `pine/bist_trend_long.pine` — BIST long-only trend takibi (Pine v6)

Grafik uzerinde calisan strategy. 16 sembolu tek tek gezip kose tablosundan
sonuclari okumak icin tasarlandi.

### Kurallar

| | |
|---|---|
| Trend filtresi | `close > EMA(200)` **ve** `EMA(200) > EMA(200)[10]` |
| Giris A — Kirilim | trend yukari **ve** `close > ta.highest(high, 20)[1]` |
| Giris B — Geri cekilme | trend yukari **ve** `close > EMA(20)` **ve** `close[1] <= EMA(20)[1]` |
| Baslangic stop | `giris - 3 * ATR(14)` |
| Takip eden stop | `(giristen sonraki en yuksek high) - 3 * ATR(14)`, `math.max` ile asla asagi inmez |
| Hedef | yok — cikis sadece stopla |
| Pozisyon | `qty = (strategy.equity * 0.005) / (giris - stop)`, `pyramiding = 0` |

`strategy()` ayarlari: komisyon yuzde bazli (input, varsayilan **%0.30**),
**slippage 0**, `process_orders_on_close=true`, `calc_on_every_tick=false`,
`initial_capital=100000`.

### Kurulum

1. TradingView → Pine Editor → dosyanin icerigini yapistir → **Add to chart**.
2. Grafigi **gunluk (1D)** periyoda al.
3. Ayarlardan giris modunu ve tarih araligini sec.

### Eski %60 / yeni %40 ayrimi

Script tarih araligini kisitlar ama gostergeleri **tum** veriden hesaplar — yani
yeni donemi test ederken EMA(200) zaten isinmis olur, donem basinda 200 barlik
olu bolge olusmaz.

Baslangic varsayilani 01.01.2015 (2005 oncesi TRY verisi kullanilamaz — asagiya
bakin). Bolme tarihini bu baslangica gore hesaplayin.

Her sembol icin:

1. Once bitis tarihini bugune al, grafikte ilk islem gorulen barin tarihini not et.
2. Toplam gun sayisinin %60'ini ekleyip bolme tarihini bul.
3. **Eski donem:** Baslangic = ilk bar, Bitis = bolme tarihi.
4. **Yeni donem:** Baslangic = bolme tarihi, Bitis = bugun.
5. Iki tabloyu ayri ayri kaydet.

`Aralik bitince aciktaki pozisyonu kapat` acik kalsin — aksi halde donem
sonunda acik kalan islem hic kapanmaz ve istatistiklere girmez.

### Kose tablosu

Sembol · islem sayisi · kazanma orani % · net getiri % · max drawdown % ·
profit factor · maks ardisik kayip · **ort kazanc (R)** · **ort kayip (R)** ·
ort kazanc/kayip orani · net kar · acik pozisyon var/yok ·
**poz. siniri** (kirpma sayisi + ham maks pozisyon).

`Ort kayip (R)` **-1.5R'i asarsa kirmizi** yanar. Beklenen deger -1R civaridir
(stop mesafesi kadar). Belirgin asiyorsa cikislar stopun cok altinda
gerceklesiyor demektir — bosluk ya da kapanista atlama var, stop mesafesi
gercek riski temsil etmiyor.

Islem sayisi **20'nin altindaysa** hucre turuncuya doner ve `(veri yetersiz)`
yazar. Max drawdown TradingView'in kendi kutusundan degil, `strategy.equity`
uzerinden secili tarih araligi icinde ayrica hesaplanir (mark-to-market, zirveye
oranla %).

### Dikkat edilecek iki nokta

**Maliyet: slippage 0, hepsi yuzde bazli komisyonda.** TradingView'de slippage
tick cinsindendir, yuzde degil. Geriye donuk duzeltme yuzunden eski fiyatlar cok
kucuk cikiyor (BTCIM 2015'te 0.08 TL) ve 5 tick = 0.05 TL fiyatin **%62'sine**
denk geliyordu. Sabit tick maliyeti fiyat seviyesine gore devasa sapma urettigi
icin slippage 0'a alindi ve tum maliyet komisyona gomuldu.

Komisyon varsayilani **%0.30**, islem BASINA — alista ve satista ayri kesilir,
yani gidis-donus maliyet **%0.60**.

**Komisyonu degistirmek icin Inputs'a bakmayin, Properties'e bakin.**
`strategy()` parametreleri `const` olmak zorundadir, `input.float()` oraya
konulamaz (`CE10123: An argument of 'input float' type was used but a
'const float' is expected`). Gerek de yok — komisyon derlemeden degistirilebilir:

```
Ayarlar (dis carki) -> Properties -> Commission
```

Oradaki deger scriptteki varsayilani ezer ve `strategy.*` degiskenlerine,
dolayisiyla kose tablosuna aninda yansir. `Commission type` alaninin **Percent**
kaldigindan emin olun. Ayni ekranda slippage'i de gorebilirsiniz — 0 olmali.

Maliyetler sadece K/Z'yi etkiler; sinyaller `close` ve hesaplanan stop
seviyesinden uretildigi icin islem **sayisi ve tarihleri** degismez. Ayni
sembolde komisyonu degistirip islem sayisinin sabit kaldigini gorerek bunu
dogrulayabilirsiniz.

Not: yuksek komisyon `Ort kayip (R)` degerini daha negatif yapar. Maliyetin R
cinsinden agirligi `gidis-donus % x fiyat / (3*ATR)` kadardir — ATR fiyata gore
kucukse (dar stop) maliyet R'nin ciddi bir kismini yer. `Ort kayip (R)`
-1.5R'i asip kirmizi yaniyorsa once bunu kontrol edin.

**Ortuk kaldirac — lot uc tavanin en kucugu.** Illikit ya da durgun veride ayni
kapanis tekrar edince TR sifir olur ve ATR cokmeye yaklasir. `risk / (3*ATR)`
formulu o barda sermayenin katlarina ulasan lot uretir; ustune bir fiyat
sureksizligi gelirse tek islemde sermayenin katlari kadar zarar yazilir.
Bu yuzden lot su ucunun **en kucugu** olarak hesaplanir:

| aday | formul | varsayilan |
|---|---|---|
| risk bazli | `sermaye * %0.5 / (3*ATR)` | — |
| pozisyon siniri | `sermaye * %20 / giris` | `Maks pozisyon buyuklugu = 20` |
| kayip tavani | `sermaye * %2 / giris` | `Tek islemde maks kayip = 2` |

Kayip tavani "en kotu ihtimalle pozisyonun tamami gider, o da sermayenin %2'sini
gecmesin" varsayimidir. **%2 tavan %20'lik pozisyon sinirindan her zaman dardir**,
yani varsayilan ayarda pozisyon siniri fiilen devre disidir; ikinci bir emniyet
kemeri olarak durur. Ikisi de `0` ile kapatilabilir.

Bunun bedeli var: 3*ATR'ye gore hesaplanan lot neredeyse her barda tavanla
kirpildigi icin **islem basina gercek risk %0.5'in belirgin altina iner**
(sentetik testte ort. %0.12) ve net getiri kuculur. Risk normalize edilmedigi
icin islemler artik esit agirlikli degildir — R istatistikleri her islemin kendi
riskine gore hesaplandigindan tutarli kalir, ama "her islem %0.5 risk"
varsayimi gecerli degildir.

Tabloda **`Poz. siniri`** satiri `<n> tavan / <n> poz, ham maks %<x>` formatinda
hangi tavanin kac kez bagladigini gosterir. `ham maks %` hicbir tavan olmasaydi
formulun actigi en buyuk pozisyondur — **asil bakilacak sayi budur**. %100'u
asiyorsa (satir turuncuya doner) o sembolde ATR'nin coktugu durgun veri bolgesi
var demektir.

**2005 oncesi TRY verisi kullanilamaz.** Redenominasyon (6 sifir atilmasi) ve
hiperenflasyon yuzunden fiyat serisi sureksiz. Bu yuzden `Tarih araligi kullan`
varsayilan olarak **acik** ve baslangic **01.01.2015**. Kapatirsaniz sembolun
tum gecmisi islenir; eski veride yukaridaki kaldirac patolojisi tetiklenir.
Pozisyon siniri bu durumda tek islem zararini sermayenin ~1 katiyla sinirlar
ama artefakti yok etmez — tarih filtresi asil koruma.

### Cikis modeli

Varsayilan `Cikis bar kapanisinda` **acik**: `close <= stop` oldugunda
`strategy.close()` ile bar kapanisinda cikilir, gun ici doldurma varsayilmaz.
Kapatirsaniz `strategy.exit(stop=...)` ile gercek bar ici stop emri kullanilir —
dolum stop fiyatindan olur ve sonuclar daha iyimser cikar. Bu modda stop emri
gonderildigi barin bir sonrasindan itibaren tetiklenir, yani giris barindan
sonraki ilk bar korumasizdir.

### Semboller

```
RODRG  OBASE  POLHO  KLSYN  ETILR  GWIND  MHRGY  BTCIM
KAREL  AVTUR  KAPLM  NTGAZ  ATAGY  TUCLK  AVPGY  DZGYO
```
