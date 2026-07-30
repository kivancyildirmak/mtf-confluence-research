"""Takım ismi normalizasyon ve eşleştirme katmanı.

Farklı kaynaklar aynı takımı farklı yazabilir:
    "Fenerbahce" / "Fenerbahçe" / "Fenerbahce SK"
    "Man United" / "Manchester United" / "Man Utd"

Amaç: kaynaklar arası isimleri ortak bir "kanonik" biçime indirgemek ve
kullanıcı seçimlerini cache'deki isimlerle güvenilir biçimde eşleştirmek.
Harici bağımlılık gerektirmemesi için basit Levenshtein tabanlı benzerlik
kullanılır (difflib da yeterli olurdu, ancak eşik davranışını kontrol etmek
istiyoruz).
"""
from __future__ import annotations

import re
import unicodedata

# Yaygın ekler/kısaltmalar — normalizasyonda atılır.
_STOPWORDS = {
    "fc", "sc", "sk", "as", "ac", "afc", "cf", "cd", "if", "bk", "sv",
    "club", "kulubu", "kulübü", "spor", "spor kulubu", "the",
}

# Elle bakım eşlemesi: sık karışan bilinen takımlar için kesin kanonik ad.
# Sol taraf normalize edilmiş biçim, sağ taraf kanonik ad.
_ALIASES = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "wolves": "wolverhampton",
    "psg": "paris sg",
    "paris saint germain": "paris sg",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "ath madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    "bayern": "bayern munich",
    "fenerbahce": "fenerbahce",
    "galatasaray": "galatasaray",
    "besiktas": "besiktas",
}


def normalize(name: str) -> str:
    """İsmi aksansız, küçük harfli, ek/durak-kelimeden arınmış biçime indirger."""
    if not name:
        return ""
    # Türkçe/aksanlı karakterleri ASCII'ye indir (Fenerbahçe -> fenerbahce)
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    # Noktalama -> boşluk
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    normalized = " ".join(tokens)
    return _ALIASES.get(normalized, normalized)


def _levenshtein(a: str, b: str) -> int:
    """Klasik dinamik programlama ile düzenleme mesafesi."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """0..1 arası benzerlik skoru (1 = birebir aynı, normalize edilmiş).

    Salt düzenleme mesafesi, takım adlarında kısa/uzun yazım çiftlerini kaçırır:
    'PSV' vs 'PSV Eindhoven' mesafesi büyüktür ama aynı takımdır. Bu yüzden
    önce kelime-kapsama kontrolü yapılır (bir adın tüm kelimeleri diğerinde
    geçiyorsa yüksek skor verilir).
    """
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    # Boşluk/tire farkları: 'Ham-Kam' vs 'HamKam'
    if na.replace(" ", "") == nb.replace(" ", ""):
        return 0.98

    # Kelime kapsama: 'psv' ⊂ 'psv eindhoven', 'bayern' ⊂ 'bayern munich'
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and (ta < tb or tb < ta):
        # Uzunluk oranına göre 0.85–0.95 arası; tam kapsama güçlü bir sinyaldir
        ratio = min(len(ta), len(tb)) / max(len(ta), len(tb))
        return 0.85 + 0.10 * ratio

    dist = _levenshtein(na, nb)
    return 1.0 - dist / max(len(na), len(nb))


def normalize_key(name: str) -> str:
    """Gruplama anahtarı: normalize edip TÜM boşlukları atar.

    'Ham-Kam' ve 'HamKam' -> 'hamkam';  'Ajax ' ve 'Ajax' -> 'ajax'.
    Aynı takımın farklı yazımlarını tek kovada toplamak için kullanılır.
    """
    return normalize(name).replace(" ", "")


def build_canonical_map(names, counts=None) -> dict:
    """Takım adı varyantlarını tek bir kanonik yazıma eşler.

    Aynı kaynakta bile bir takım birden çok yazımla geçebilir
    (ör. 'Ajax' / 'Ajax ' veya 'Ham-Kam' / 'HamKam'). Bunlar ayrı takım
    sayılırsa maç geçmişi bölünür ve her iki kaydın da güç tahmini bozulur.

    Args:
        names: ham takım adları.
        counts: opsiyonel {ad: maç_sayısı} — en sık kullanılan yazım seçilir.

    Returns:
        {ham_ad: kanonik_ad} sözlüğü (baştaki/sondaki boşluklar da temizlenir).
    """
    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if n is None:
            continue
        groups[normalize_key(str(n))].append(str(n))

    counts = counts or {}
    mapping: dict[str, str] = {}
    for variants in groups.values():
        # En sık kullanılan; eşitlikte daha uzun, sonra alfabetik (deterministik)
        best = sorted(
            variants,
            key=lambda v: (-counts.get(v, 0), -len(v.strip()), v),
        )[0].strip()
        for v in variants:
            mapping[v] = best
    return mapping


def unify_team_names(df, home_col: str = "home_team", away_col: str = "away_team"):
    """DataFrame'deki takım adlarını kanonik yazıma indirger (kopya döndürür)."""
    if df is None or len(df) == 0:
        return df
    if home_col not in df.columns or away_col not in df.columns:
        return df

    counts = {}
    for col in (home_col, away_col):
        for name, n in df[col].value_counts().items():
            counts[name] = counts.get(name, 0) + int(n)

    mapping = build_canonical_map(counts.keys(), counts)
    out = df.copy()
    out[home_col] = out[home_col].map(lambda x: mapping.get(x, str(x).strip() if x is not None else x))
    out[away_col] = out[away_col].map(lambda x: mapping.get(x, str(x).strip() if x is not None else x))
    return out


def best_match(query: str, candidates: list[str], threshold: float = 0.6):
    """`candidates` içinden `query`'e en çok benzeyen ismi döndürür.

    Returns:
        (eşleşen_isim, skor) veya eşik altındaysa (None, en_iyi_skor).
    """
    best_name = None
    best_score = 0.0
    for cand in candidates:
        s = similarity(query, cand)
        if s > best_score:
            best_score, best_name = s, cand
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score
