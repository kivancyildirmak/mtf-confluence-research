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


def test_similarity_short_vs_long_name():
    """Kısa/uzun yazım aynı takımı göstermeli (kelime kapsama)."""
    assert nm.similarity("PSV", "PSV Eindhoven") > 0.8
    assert nm.similarity("Bayern", "Bayern Munich") > 0.8
    assert nm.similarity("Sporting", "Sporting Braga") > 0.8


def test_similarity_hyphen_vs_concatenated():
    assert nm.similarity("Ham-Kam", "HamKam") > 0.9


def test_containment_does_not_merge_rival_clubs():
    """Aynı şehrin farklı kulüpleri birleşmemeli."""
    assert nm.similarity("Manchester United", "Manchester City") < 0.8
    match, _ = nm.best_match(
        "Manchester City", ["Manchester United", "Manchester City"], threshold=0.75
    )
    assert match == "Manchester City"


def test_best_match_finds_closest():
    cands = ["Fenerbahce", "Galatasaray", "Besiktas"]
    match, score = nm.best_match("Fenerbahçe", cands)
    assert match == "Fenerbahce"
    assert score == 1.0


def test_best_match_below_threshold_returns_none():
    cands = ["Fenerbahce", "Galatasaray"]
    match, score = nm.best_match("Real Madrid", cands, threshold=0.6)
    assert match is None


# --------------------------------------------------------------------------- #
# Aynı takımın farklı yazımlarını birleştirme
# --------------------------------------------------------------------------- #
def test_normalize_key_ignores_spacing_and_punctuation():
    assert nm.normalize_key("Ham-Kam") == nm.normalize_key("HamKam")
    assert nm.normalize_key("Ajax ") == nm.normalize_key("Ajax")
    assert nm.normalize_key("Nott'm Forest") == nm.normalize_key("Nottm Forest")


def test_build_canonical_map_picks_most_frequent():
    names = ["Ajax", "Ajax "]
    counts = {"Ajax": 100, "Ajax ": 3}
    mapping = nm.build_canonical_map(names, counts)
    assert mapping["Ajax"] == "Ajax"
    assert mapping["Ajax "] == "Ajax"      # nadir varyant kanonike bağlanır


def test_build_canonical_map_strips_whitespace():
    mapping = nm.build_canonical_map(["Feyenoord "], {"Feyenoord ": 5})
    assert mapping["Feyenoord "] == "Feyenoord"


def test_unify_team_names_merges_variants():
    import pandas as pd
    df = pd.DataFrame({
        "home_team": ["Ajax", "Ajax ", "HamKam", "Ham-Kam"],
        "away_team": ["PSV", "PSV", "Molde", "Molde"],
        "home_goals": [1, 2, 0, 1],
        "away_goals": [0, 1, 2, 1],
    })
    out = nm.unify_team_names(df)
    # Dört ayrı yazım iki takıma indirgenmeli
    assert set(out["home_team"]) == {"Ajax", "HamKam"} or \
           set(out["home_team"]) == {"Ajax", "Ham-Kam"}
    assert len(set(out["home_team"])) == 2


def test_unify_handles_empty_dataframe():
    import pandas as pd
    empty = pd.DataFrame(columns=["home_team", "away_team"])
    assert len(nm.unify_team_names(empty)) == 0
