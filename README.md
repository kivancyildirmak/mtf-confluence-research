# ⚽ Futbol Maç Tahmin Uygulaması (Dixon-Coles / Poisson)

Geçmiş maç verilerini **kendisi indiren**, Dixon-Coles genişletmeli Poisson
istatistik modeliyle takım güçlerini öğrenen ve bir maç için olasılık tahmini
üreten masaüstü/veri uygulaması.

> ⚠️ **Bu araç garanti sonuç vermez; yalnızca istatistiksel olasılık üretir.**
> Ayrıntılı yasal/etik uyarı için [aşağıya](#-yasal--etik-uyarı) bakın.

---

## Özellikler

- **Otomatik veri:** [football-data.co.uk](https://www.football-data.co.uk/)
  üzerinden lig CSV'lerini (Süper Lig, Premier Lig, La Liga, Bundesliga,
  Serie A, Ligue 1 vb.) çoklu sezon halinde indirir, yerel **SQLite cache**'e
  yazar. Her açılışta baştan indirmez — yalnızca eksik/yeni maçları ekler.
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
URL kalıbı: `https://www.football-data.co.uk/mmz4281/{SEZON}/{KOD}.csv`
(ör. `.../mmz4281/2324/E0.csv`). Uygulama son N sezonu otomatik indirip birleştirir.
Elle CSV indirmeniz/yüklemeniz **gerekmez** (indirme başarısızsa elle yükleme yedeği vardır).

### 2) Sakatlık / kadro — resmi API (opsiyonel)
[API-Football (api-sports.io)](https://www.api-sports.io/) gibi ücretsiz kotalı
bir API'nin `injuries` endpoint'i desteklenir. Anahtar **Kadro/Sakatlık**
ekranından girilir ve yalnızca yerelde (`~/.football_predictor/settings.json`)
saklanır. Anahtar yoksa elle güç çarpanı girebilirsiniz.

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
