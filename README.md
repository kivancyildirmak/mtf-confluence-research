# ⚽ Futbol Maç Tahmin Uygulaması (Dixon-Coles / Poisson)

Geçmiş maç verilerini **kendisi indiren**, Dixon-Coles genişletmeli Poisson
istatistik modeliyle takım güçlerini öğrenen ve bir maç için olasılık tahmini
üreten masaüstü/veri uygulaması.

> ⚠️ **Bu araç garanti sonuç vermez; yalnızca istatistiksel olasılık üretir.**
> Ayrıntılı yasal/etik uyarı için [aşağıya](#-yasal--etik-uyarı) bakın.

---

## Özellikler

- **Otomatik veri:** [football-data.co.uk](https://www.football-data.co.uk/)
  üzerinden **32 lig** (büyük Avrupa ligleri + İskandinavya, Polonya, Belçika,
  Avusturya, İsviçre, ABD, Brezilya…) indirir, yerel **SQLite cache**'e yazar.
  Her açılışta baştan indirmez — yalnızca eksik/yeni maçları ekler.
  Desteklenen ligler için [aşağıya](#desteklenen-ligler) bakın.
- **Dixon-Coles modeli:**
  - Her takım için hücum/savunma katsayısı,
  - veriden öğrenilen ev sahibi avantajı,
  - düşük skorlar için `rho` düzeltmesi,
  - üstel zaman ağırlığı (ayarlanabilir yarı-ömür, varsayılan **180 gün**),
  - `scipy.optimize.minimize` ile maksimum olabilirlik.
- **Çıktılar:** 1/X/2, KG Var/Yok, Üst/Alt 2.5, en olası skor, beklenen goller,
  skor olasılık matrisi.
- **Kadro/Sakatlık:** opsiyonel API-Football anahtarı **veya** elle "güç
  çarpanı" girişi (kaba yaklaşım olduğu arayüzde belirtilir).
- **Backtest:** walk-forward (veri sızıntısız) test; doğruluk, log-loss, Brier;
  naif temellerle grafik karşılaştırma ve otomatik yorum.
- **Olasılık kalibrasyonu:** model yüksek olasılıklarda fazla iddialıdır;
  backtest kayıtlarından sıcaklık ölçekleme (temperature scaling) öğrenilerek
  gösterilen yüzdeler gerçekleşme oranına yaklaştırılır ([ayrıntı](#olasılık-kalibrasyonu)).
- **Değer analizi (bonus):** model olasılıkları vs oranların ima olasılıkları;
  net risk uyarısıyla.

---

## Mimari

```
football-predictor/
├── app.py                  # Streamlit giriş noktası (ekran yönlendirme)
├── run_desktop.py          # PyInstaller paketi için başlatıcı
├── football_predictor.spec # PyInstaller spec
├── requirements.txt
├── fpredict/               # UI'dan bağımsız çekirdek (test edilebilir)
│   ├── config.py           # lig kodları, yollar, varsayılanlar
│   ├── db.py               # SQLite cache
│   ├── name_matching.py    # takım ismi normalizasyon/eşleştirme
│   ├── data_fetch.py       # football-data.co.uk indirici + CSV ayrıştırma
│   ├── model.py            # Dixon-Coles Poisson modeli
│   ├── squad_adjust.py     # sakatlık API + elle güç çarpanları
│   ├── backtest.py         # walk-forward backtest
│   └── value.py            # değerli bahis karşılaştırması
├── ui/                     # Streamlit ekranları
│   ├── main_screen.py      # tahmin
│   ├── data_screen.py      # veri güncelleme
│   ├── squad_screen.py     # kadro/sakatlık
│   ├── backtest_screen.py  # backtest
│   └── value_screen.py     # değer
├── tests/                  # pytest birim testleri
└── sample_data/            # çevrimdışı test için sentetik CSV
```

**Neden Streamlit?** Bu bir veri/analiz uygulaması; çok ekranlı arayüz, tablo,
grafik ve form girişleri Streamlit ile çok az kodla ve temiz çıkar. Çekirdek
mantık (`fpredict/`) arayüzden tamamen ayrıdır; istenirse PySide6 gibi native
bir kabuğa taşınabilir.

---

## Hızlı Başlangıç (Windows — komut yazmadan)

Python 3.11+ kurulu olsun ([python.org](https://www.python.org/downloads/),
kurulumda **"Add python.exe to PATH"** kutusunu işaretleyin).

1. **`Baslat.bat`** dosyasına **çift tıklayın.** Hepsi bu.
   - Eksik paketleri ilk açılışta kendisi kurar
   - Streamlit'in ilk çalıştırma e-posta sorusunu atlar
   - Tarayıcıyı otomatik açar
2. Her seferinde daha da kolay olsun isterseniz **`Masaustu_Kisayolu_Olustur.bat`**
   dosyasını bir kez çalıştırın — masaüstüne kısayol ekler.

Kapatmak için açılan siyah pencereyi kapatın.

**macOS / Linux:** `chmod +x baslat.sh` (bir kez) → `./baslat.sh`

---

## Kurulum (kaynaktan çalıştırma)

Python 3.11+ gerekir.

```bash
# 1) Depoyu alın ve dizine girin
cd football-predictor

# 2) (Önerilir) sanal ortam
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3) Bağımlılıklar
pip install -r requirements.txt

# 4) Çalıştırın
streamlit run app.py
```

Tarayıcıda otomatik açılır (varsayılan `http://localhost:8501`).

### İlk kullanım

1. **Veri** sekmesine gidin → bir lig seçin → **Verileri Güncelle**'ye basın.
   (İndirme başarısız olursa CSV'yi elle yükleyebilirsiniz — yedek seçenek arayüzdedir.)
2. **Ana (Tahmin)** sekmesinde ligi ve ev/deplasman takımını seçin → **Tahmin Et**.
3. İsterseniz **Kadro/Sakatlık** sekmesinden güç çarpanı girin.
4. **Backtest** ve **Değer** sekmeleriyle modeli değerlendirin.

---

## Testler

```bash
pip install pytest
pytest -q
```

Testler ağ gerektirmez (sentetik veri + `sample_data/`). Model matematiği
(parametre geri kazanımı, olasılık toplamı, tau düzeltmesi, zaman ağırlığı),
isim eşleştirme ve backtest/değer mantığı kapsanır.

---

## Paketleme (tek dosyalık masaüstü uygulaması)

Streamlit + PyInstaller özel yapılandırma gerektirir; bu yüzden hazır bir
`.spec` ve başlatıcı (`run_desktop.py`) sağlanmıştır.

```bash
pip install pyinstaller
pyinstaller football_predictor.spec
```

Çıktı: `dist/FootballPredictor` (Linux/macOS) veya
`dist/FootballPredictor.exe` (Windows). Çalıştırıldığında bir konsol açılır,
Streamlit sunucusu başlar ve tarayıcı otomatik açılır.

**Notlar:**
- Konsol penceresi bilerek açık bırakıldı (`console=True`) — sunucu portunu ve
  olası hataları görmek için. İsterseniz `.spec` içinde `False` yapın.
- `upx=True` derlemeyi küçültür; UPX kurulu değilse otomatik atlanır.
- Antivirüs, imzasız PyInstaller exe'lerini işaretleyebilir; bu yaygın bir
  yanlış-pozitiftir.
- Yerel cache ve ayarlar kullanıcının ev dizininde
  `~/.football_predictor/` altında tutulur (paket klasörü salt-okunur olsa bile
  sorun çıkmaz).

---

## Veri Kaynakları

### 1) Geçmiş sonuçlar + oranlar — football-data.co.uk

Site verileri **iki farklı biçimde** yayınlar; uygulama ikisini de destekler ve
her lig için doğru olanı otomatik seçer:

| Biçim | URL | Kapsam | Sütunlar |
|---|---|---|---|
| `main` | `mmz4281/{SEZON}/{KOD}.csv` | Büyük Avrupa ligleri | `HomeTeam, FTHG, FTAG, FTR, B365H…` |
| `extra` | `new/{KOD}.csv` | İskandinavya, Polonya, ABD, Brezilya… | `Home, HG, AG, Res, AvgH…` |

`main` biçimde her sezon ayrı dosyadır (son N sezon indirilir); `extra` biçimde
tüm sezonlar tek dosyadadır (tam geçmiş indirilir, sezon seçimi devre dışıdır).
Elle CSV yüklerken biçim **otomatik algılanır**.

#### Erişim engellenirse — arşiv kaynağı (tüm ligler)

football-data.co.uk bazı ülkelerde ağ seviyesinde engellidir. **Veri** ekranındaki
*"Yalnızca arşiv (GitHub)"* seçeneği bu durumda **38 ligin tamamını** GitHub
üzerinden indirir (2000-2025, ~230.000 maç, oran sütunları dahil).

> ⚠️ **Arşiv canlı değildir.** Son maç tarihi ligden lige değişir: büyük Avrupa
> ligleri ~Mayıs 2025, İskandinav/Polonya ligleri ~Aralık 2024. Uygulama bunu
> güncelleme sonrası açıkça uyarı olarak gösterir. Yeni sezon maçlarında
> tahminlerin güvenilirliği düşer; güncel veri için asıl kaynağa erişim veya
> API-Football anahtarı gerekir.

#### Erişim engellenirse — yedek ayna

Bazı ağlarda (İSS/DNS engeli, kurumsal filtre, antivirüsün HTTPS taraması)
football-data.co.uk'a erişilemez. Belirtiler: `CERTIFICATE_VERIFY_FAILED`,
bağlantı zaman aşımı veya `ConnectionResetError 10054`.

Uygulama bu durumda:
1. her indirmeyi **3 kez** artan beklemeyle tekrar dener,
2. **GitHub aynasına** düşer (aynı sütun düzeni, oran sütunları yok),
3. başarısız olursa **teşhis + çözüm adımları** gösterir (DNS değiştirme,
   antivirüs HTTPS taraması, VPN, elle CSV yükleme).

> Ayna yalnızca **Premier Lig, La Liga, Serie A, Bundesliga, Ligue 1**'i kapsar;
> bu ligler engelden bağımsız çalışır. Diğer ligler için asıl kaynağa erişim
> gerekir. Aynada oran sütunu olmadığından *Değer* ekranı ayna verisiyle çalışmaz.

#### Desteklenen ligler

**`main` biçim:** Türkiye Süper Lig (T1), İngiltere Premier Lig / Championship
(E0, E1), İskoçya Premiership (SC0), İspanya La Liga 1-2 (SP1, SP2), Almanya
Bundesliga 1-2 (D1, D2), İtalya Serie A-B (I1, I2), Fransa Ligue 1-2 (F1, F2),
Hollanda Eredivisie (N1), **Belçika Jupiler Pro Lig (B1)**, Portekiz Primeira
Liga (P1), Yunanistan Super Lig (G1).

**`extra` biçim:** **İsveç Allsvenskan (SWE)**, **Norveç Eliteserien (NOR)**,
**Danimarka Superliga (DNK)**, Finlandiya (FIN), **Polonya Ekstraklasa (POL)**,
Avusturya (AUT), İsviçre (SWZ), İrlanda (IRL), Romanya (ROU), Rusya (RUS),
ABD MLS (USA), Meksika (MEX), Brezilya (BRA), Arjantin (ARG), Japonya (JPN),
Çin (CHN).

> **Spor Toto listeleri hakkında:** Haftalık kuponlarda sık geçen İskandinav,
> Belçika ve Polonya ligleri yukarıda kapsanmıştır. Ancak listelerde **kupa
> maçları** (ör. Şampiyonlar Ligi elemeleri) ve **alt lig takımları** da yer
> alabilir; football-data.co.uk yalnızca lig maçlarını yayınladığı için bu tür
> eşleşmeler modelde bulunmaz.

### 2) API-Football — güncel sezon verisi (önerilen yedek)

football-data.co.uk erişilemiyorsa ve **güncel** veri gerekiyorsa
[API-Football (api-sports.io)](https://www.api-sports.io/) kullanılabilir.
Ücretsiz plan **günde 100 istek** verir; bir ligin bir sezonu **tek istek**
harcadığından bu fazlasıyla yeterlidir.

**İki kanal desteklenir** — aynı API, farklı alan adları. Biri ağınızda
engelliyse diğerini seçebilirsiniz (veri ve kullanım birebir aynıdır):

| Kanal | Ana makine | Kayıt |
|---|---|---|
| Doğrudan | `v3.football.api-sports.io` | dashboard.api-football.com/register |
| RapidAPI | `api-football-v1.p.rapidapi.com` | rapidapi.com/api-sports/api/api-football |

> Anahtarlar kanala özeldir: RapidAPI anahtarı doğrudan kanalda çalışmaz.

**Kurulum:**
1. **Veri** ekranı → *Veri kaynağı: API-Football* → **API kanalı**'nı seçin.
2. Seçtiğiniz kanaldan ücretsiz kaydolup anahtarınızı kopyalayın.
3. Anahtarı yapıştırıp kaydedin.
4. **🔌 Bağlantıyı Test Et** ile anahtarı ve kalan kotayı doğrulayın.
5. Ligi seçip **Verileri Güncelle**'ye basın.

**Lig ID doğrulama:** Uygulamada 32 lig için yerleşik API lig ID'si vardır.
Bir ID yanlış/eskimişse API 0 maç döndürür; bu durumda **🔎 Lig ID ara / düzelt**
aracıyla ligi adıyla arayıp doğru ID'yi seçebilirsiniz. Seçim
`settings.json` içine kalıcı yazılır ve yerleşik değeri geçersiz kılar.

**Sınırlar:**
- Ücretsiz planda **bahis oranı yoktur** → *Değer* ekranı API verisiyle çalışmaz
  (Tahmin ve Backtest normal çalışır).
- API'den gelen takım adları, cache'te zaten bulunan adlara otomatik çapalanır
  (aynı takımın iki yazımla bölünmesini önlemek için).

### 3) Sakatlık / kadro — resmi API (opsiyonel)
Aynı API anahtarıyla **Kadro/Sakatlık** ekranından bir takımın sakat/cezalı
oyuncuları çekilebilir. Takım ID'si lig kadrosundan otomatik çözülür (isim
eşleştirmesiyle); elle ID girmeniz gerekmez. Anahtar yalnızca yerelde
(`~/.football_predictor/settings.json`) saklanır, hiçbir yere gönderilmez.

Çekilen liste **bilgilendirme amaçlıdır**: hangi oyuncunun ne kadar kritik
olduğunu uygulama bilemez, bu yüzden modele etkiyi siz güç çarpanıyla
verirsiniz. Anahtar yoksa çarpanı doğrudan elle girebilirsiniz.

> Rastgele sitelerden ham scraping **yapılmaz** — yalnızca resmi/izinli API'ler
> ve açık veri kullanılır. Yalnızca kullanım şartlarına uygun kaynakları kullanın.

---

## İstatistik Modeli — kısa açıklama

Ev sahibi *i*, deplasman *j* için beklenen goller:

```
log(λ_ev)        = hücum[i] − savunma[j] + ev_avantajı
log(μ_deplasman) = hücum[j] − savunma[i]
```

Goller Poisson dağılır; düşük skor hücreleri (0-0, 1-0, 0-1, 1-1) için
Dixon-Coles `τ(x,y)` düzeltmesi uygulanır. Her maç, güncelliğine göre üstel
azalan bir ağırlık alır: `w = 2^(−yaş_gün / yarı_ömür)`. Parametreler ağırlıklı
maksimum olabilirlikle (`L-BFGS-B`) bulunur; hücum ortalaması 0'a sabitlenerek
konum belirsizliği kırılır.

---

## Olasılık Kalibrasyonu

Ham Dixon-Coles modeli düşük/orta olasılıklarda iyi kalibredir ama **yüksek
olasılıklarda fazla iddialıdır**. Norveç Eliteserien'de 2.676 maçlık
walk-forward backtest ile ölçülen gerçek davranış:

| Model ne dedi | Maç | Gerçekleşme (ham) | Gerçekleşme (kalibre) |
|---|---|---|---|
| %0–40 | 354 | %37.3 (fark 0.7) | fark **0.5** |
| %40–50 | 925 | %43.0 (fark 1.8) | fark **1.4** |
| %50–60 | 711 | %52.7 (fark 2.0) | fark **0.9** |
| %60–70 | 403 | %61.8 (fark 2.5) | fark **0.4** |
| **%70+** | 283 | %69.3 (fark **7.5**) | fark **2.8** |

**Yöntem:** sıcaklık ölçekleme — olasılıklar `p^(1/T)` ile yumuşatılır ve
normalize edilir. T, backtest kayıtlarından log-olabilirlik en büyüklenerek
bulunur. Tek parametreli olduğu için aşırı uyum riski çok düşüktür ve
tahminlerin **sıralamasını değiştirmez**, yalnızca güven düzeyini düzeltir.

Backtest kayıtları örneklem-dışıdır (her maç yalnızca kendisinden önceki
verilerle tahmin edilir), bu yüzden kalibrasyon eğitimi için uygundur.

**Pazar başına ayrı sıcaklık öğrenilir.** Ölçümler, aşırı güvenin asıl olarak
gol pazarlarında olduğunu gösterdi:

| Pazar | Öğrenilen T | Etki (örnek maç) |
|---|---|---|
| 1X2 | 1.19 | %87.6 → %82.2 |
| Üst/Alt 2.5 | **1.81** | %82.2 → **%70.0** |
| KG Var/Yok | 1.63 | %54.4 → %52.7 |

**Kullanımı:** *Backtest* ekranında testi çalıştırın → **Kalibrasyonu Kaydet ve
Uygula**. Kayıt lig bazındadır (`settings.json`) ve *Ana (Tahmin)* ekranındaki
olasılıklara otomatik uygulanır. Beklenen gol sayıları ve en olası skor ham
model çıktısı olarak kalır.

**Neden lig bazında?** Ev sahibi avantajı, beraberlik oranı ve gol dağılımı
ligden lige değişir; dolayısıyla modelin aşırı güven miktarı da değişir. Bir
ligde öğrenilen sıcaklığı diğerine uygulamak yanlış olur.

**Toplu kalibrasyon:** Her ligi elle kalibre etmemek için *Backtest* ekranındaki
**⚙️ Tüm ligleri toplu kalibre et** bölümünü kullanın. Cache'teki her lig için
backtest çalıştırıp kalibrasyonu kaydeder, sonuçları tablo halinde özetler.
Verisi yetersiz ligler atlanır; bir ligdeki hata diğerlerini durdurmaz.
(Lig başına birkaç dakika sürebilir.)

---

## 🔒 Yasal / Etik Uyarı

- **Garanti yok:** Uygulama garanti sonuç vermez; yalnızca istatistiksel
  olasılık üretir. Geçmiş performans gelecek sonuçları garanti etmez.
- **Bahis risk içerir:** Kaybetmeyi göze alamayacağınız parayı **riske atmayın**.
  Bahis, finansal ve psikolojik zarara yol açabilir.
- **Değer analizi kazanç garantisi değildir:** "Değerli" etiketi yalnızca modelin
  görüşüdür; model yanlış olabilir, piyasa oranları verimli olabilir.
- **Veri kullanımı:** Yalnızca kullanım şartlarına uygun veri kaynakları
  kullanın. Bu proje eğitim ve araştırma amaçlıdır.

### 📞 Bahis Bağımlılığı Destek Hattı

- **Türkiye:** **ALO 191** — Uyuşturucu ve Bağımlılıkla Mücadele Danışmanlık;
  **YEDAM / Yeşilay Danışmanlık Merkezi** (yesilay.org.tr).
- **Uluslararası:** Ülkenizdeki yerel bağımlılık destek hattına başvurun.

Kumar/bahisin bir eğlence değil sorun haline geldiğini düşünüyorsanız yardım
istemek güçlü bir adımdır.

---

## Lisans

Eğitim ve araştırma amaçlı örnek proje.
