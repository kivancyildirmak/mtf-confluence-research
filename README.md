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
| Pozisyon | `qty = math.floor(strategy.equity / close)` — full capital compounding, `pyramiding = 0` |

`strategy()` ayarlari: komisyon yuzde bazli **%0.30** (Properties'ten
degistirilebilir), **slippage 0**, `process_orders_on_close=true`,
`calc_on_every_tick=false`, `initial_capital=100000`.

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
**sermaye kullanimi** (min - maks %).

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

**Pozisyon: full capital compounding.** Her giriste mevcut sermayenin tamami
kullanilir:

```
qty = math.floor(strategy.equity / close)
```

Giris aninda pozisyon flat oldugu icin `strategy.equity` = 100.000 + kapanmis
islemlerin net K/Z'si; yani karlar sonraki islemlerde **bilesik** olarak devreder.
`pyramiding = 0`, ayni anda tek pozisyon.

`Tam lot (floor) kullan` varsayilan **acik** — kesirli lot alinmaz, artan bakiye
nakit kalir. Kapatirsaniz kesirli lot kullanilir ve sermayenin tam %100'u calisir.

**Risk bazli lot, pozisyon siniri ve kayip tavani kaldirildi.** 3*ATR stop hala
cikisi belirliyor ama artik **lotu sinirlamiyor**. Sonuclari yorumlarken bunun
anlami: tek islemdeki zarar, fiyatin giristen cikisa dususu kadardir — stop
bosluk yuzunden atlanirsa daha da fazlasi. Drawdown'lar risk bazli surume gore
cok daha buyuk cikacaktir; sentetik testte tek bir -%99'luk bar hesabi
sifirliyor. Bu modelin dogasi, hata degil.

`Ort kayip (R)` satiri korundu — R paydasi artik `lot x 3*ATR`, yani stop
mesafesinin lot cinsinden karsiligi. ATR'nin coktugu barlarda bu payda cok
kuculdugu icin R degeri devasa negatiflere gidebilir; **-1.5R kirmizisi** bu
durumda dogru uyariyi verir.

Tabloda **`Sermaye kullanimi`** satiri girislerde sermayenin yuzde kacinin
fiilen pozisyona girdigini `min - maks` araligi olarak gosterir. Tam lot
modunda %100'un biraz altinda olmasi normaldir. **%90'in altina duserse satir
turuncuya doner** — fiyat sermayeye gore yuksek demektir (az sayida lot
alinabiliyor), o sembolde bilesiklenme beklenenden yavas ilerler.

**2005 oncesi TRY verisi kullanilamaz.** Redenominasyon (6 sifir atilmasi) ve
hiperenflasyon yuzunden fiyat serisi sureksiz. Bu yuzden `Tarih araligi kullan`
varsayilan olarak **acik** ve baslangic **01.01.2015**. Kapatirsaniz sembolun
tum gecmisi islenir. Full capital compounding modelinde pozisyon siniri
olmadigi icin **tarih filtresi tek korumadir**: eski veride tek bir
sureksizlik bari hesabi sifirlar.

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
