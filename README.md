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

`strategy()` ayarlari sabit: komisyon %0.05 (percent), slippage 5,
`process_orders_on_close=true`, `calc_on_every_tick=false`, `initial_capital=100000`.

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
profit factor · maks ardisik kayip · ortalama kazanc/kayip (R ve oran) ·
net kar · acik pozisyon var/yok · **poz. siniri** (kirpma/asim sayisi).

Islem sayisi **20'nin altindaysa** hucre turuncuya doner ve `(veri yetersiz)`
yazar. Max drawdown TradingView'in kendi kutusundan degil, `strategy.equity`
uzerinden secili tarih araligi icinde ayrica hesaplanir (mark-to-market, zirveye
oranla %).

### Dikkat edilecek iki nokta

**Slippage tick cinsindendir, yuzde degil.** BIST'te mintick genelde 0.01 TL, yani
5 tick = 0.05 TL. Bu 20 TL'lik hissede ~%0.25, 100 TL'lik hissede ~%0.05 eder —
sembol karsilastirirken akilda tutun. Kayma ve komisyon sadece K/Z'yi etkiler;
sinyaller `close` ve hesaplanan stop seviyesinden uretildigi icin islem
**sayisi ve tarihleri** bu ayarlardan etkilenmez.

**Ortuk kaldirac — varsayilan sinir %100.** Illikit ya da durgun veride ayni
kapanis tekrar edince TR sifir olur ve ATR cokede yaklasir. `risk / (3*ATR)`
formulu o barda sermayenin katlarina ulasan lot uretir; ustune bir fiyat
sureksizligi gelirse tek islemde sermayenin katlari kadar zarar yazilir.
Bu yuzden `Maks pozisyon buyuklugu` varsayilani **100** (pozisyon degeri
sermayeyi asamaz). `0` yazarak sinirsiz hale getirebilirsiniz — spesifikasyonun
ham hali budur ama ne yaptiginizi bilerek kullanin.

Sinirin devreye girip girmedigi tabloda **`Poz. siniri`** satirinda gorunur:
kac islemde kirpildigi, kac islemde pozisyon sermayeyi astigi ve sinir
olmasaydi ulasilacak en buyuk pozisyon (`ham maks %`). Bu deger %100'un cok
uzerindeyse sembolun verisinde durgun/bozuk bolge var demektir.

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
