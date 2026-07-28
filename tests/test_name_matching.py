"""İsim normalizasyon ve eşleştirme testleri."""
from fpredict import name_matching as nm


def test_normalize_turkish_accents():
    assert nm.normalize("Fenerbahçe") == "fenerbahce"
    assert nm.normalize("Beşiktaş") == "besiktas"
    assert nm.normalize("Başakşehir") == "basaksehir"


def test_normalize_strips_suffixes():
    assert nm.normalize("Trabzonspor FC") == "trabzonspor"
    assert nm.normalize("Galatasaray SK") == "galatasaray"


def test_alias_resolution():
    assert nm.normalize("Man United") == "manchester united"
    assert nm.normalize("Man Utd") == "manchester united"
    assert nm.normalize("PSG") == "paris sg"


def test_similarity_identical_after_normalization():
    assert nm.similarity("Fenerbahce", "Fenerbahçe") == 1.0
    assert nm.similarity("Man United", "Manchester United") == 1.0


def test_similarity_different_teams_low():
    assert nm.similarity("Fenerbahce", "Galatasaray") < 0.5


def test_best_match_finds_closest():
    cands = ["Fenerbahce", "Galatasaray", "Besiktas"]
    match, score = nm.best_match("Fenerbahçe", cands)
    assert match == "Fenerbahce"
    assert score == 1.0


def test_best_match_below_threshold_returns_none():
    cands = ["Fenerbahce", "Galatasaray"]
    match, score = nm.best_match("Real Madrid", cands, threshold=0.6)
    assert match is None
