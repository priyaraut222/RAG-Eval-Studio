from backend.evaluation.metrics.text_utils import coverage_ratio, token_overlap_f1, tokenize


def test_tokenize_lowercases_and_strips_stopwords():
    tokens = tokenize("The Eiffel Tower is in Paris")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "eiffel" in tokens
    assert "tower" in tokens
    assert "paris" in tokens


def test_tokenize_without_dropping_stopwords():
    tokens = tokenize("The Eiffel Tower", drop_stopwords=False)
    assert "the" in tokens


def test_token_overlap_f1_identical_text_is_one():
    assert token_overlap_f1("Gustave Eiffel designed the tower", "Gustave Eiffel designed the tower") == 1.0


def test_token_overlap_f1_disjoint_text_is_zero():
    assert token_overlap_f1("apples and oranges", "quantum physics research") == 0.0


def test_token_overlap_f1_empty_input_is_zero():
    assert token_overlap_f1("", "something") == 0.0
    assert token_overlap_f1("something", "") == 0.0


def test_token_overlap_f1_partial_overlap_between_zero_and_one():
    score = token_overlap_f1("the tower was designed by Gustave Eiffel", "Gustave Eiffel built famous structures")
    assert 0.0 < score < 1.0


def test_coverage_ratio_full_coverage_is_one():
    claim = "Gustave Eiffel designed the tower"
    source = "The tower in Paris was designed by Gustave Eiffel in 1889."
    assert coverage_ratio(claim, source) == 1.0


def test_coverage_ratio_partial_coverage():
    claim = "Gustave Eiffel designed the tower in 1889"
    source = "Gustave Eiffel designed the tower."
    score = coverage_ratio(claim, source)
    assert 0.0 < score < 1.0


def test_coverage_ratio_no_overlap_is_zero():
    assert coverage_ratio("quantum physics", "apple pie recipe") == 0.0


def test_coverage_ratio_empty_claim_is_zero():
    assert coverage_ratio("", "some source text") == 0.0


def test_coverage_ratio_is_directional_not_symmetric():
    # A short claim fully contained in a long source scores 1.0,
    # but the reverse (long claim vs short source) should score lower.
    short = "Gustave Eiffel"
    long_text = "Gustave Eiffel was a French civil engineer who designed many famous structures across Europe."
    assert coverage_ratio(short, long_text) == 1.0
    assert coverage_ratio(long_text, short) < 1.0
