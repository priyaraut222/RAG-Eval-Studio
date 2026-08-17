from backend.evaluation.metrics.text_utils import coverage_ratio, token_overlap_f1, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize("The Eiffel Tower is in Paris")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "eiffel" in tokens
    assert "tower" in tokens
    assert "paris" in tokens


def test_tokenize_keep_stopwords_when_disabled():
    tokens = tokenize("The Eiffel Tower", drop_stopwords=False)
    assert "the" in tokens


def test_coverage_ratio_full_containment_is_one():
    assert coverage_ratio("Paris France", "The city of Paris is in France, Europe.") == 1.0


def test_coverage_ratio_no_overlap_is_zero():
    assert coverage_ratio("mountain climbing gear", "bakery bread recipe") == 0.0


def test_coverage_ratio_partial_overlap_between_zero_and_one():
    score = coverage_ratio("Paris London Tokyo", "Paris is a city.")
    assert 0.0 < score < 1.0


def test_coverage_ratio_empty_claim_is_zero():
    assert coverage_ratio("", "some source text") == 0.0


def test_coverage_ratio_is_directional_not_symmetric():
    # All of "Paris" is covered by the longer source, but not vice versa.
    forward = coverage_ratio("Paris", "Paris is the capital of France and a major European city")
    backward = coverage_ratio("Paris is the capital of France and a major European city", "Paris")
    assert forward == 1.0
    assert backward < forward


def test_token_overlap_f1_identical_text_is_one():
    assert token_overlap_f1("Gustave Eiffel designed the tower", "Gustave Eiffel designed the tower") == 1.0


def test_token_overlap_f1_no_overlap_is_zero():
    assert token_overlap_f1("mountain climbing", "bakery bread") == 0.0


def test_token_overlap_f1_empty_inputs_are_zero():
    assert token_overlap_f1("", "something") == 0.0
    assert token_overlap_f1("something", "") == 0.0


def test_token_overlap_f1_is_symmetric():
    a = "Gustave Eiffel designed the tower in 1889"
    b = "The tower was designed by Eiffel"
    assert token_overlap_f1(a, b) == token_overlap_f1(b, a)
