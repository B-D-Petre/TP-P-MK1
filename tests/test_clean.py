from poc_trustpilot.Pipeline.clean import canonicalize_brand, is_low_content, merge_title, normalize_text, truncate_text


def test_normalize_decodes_entities_and_typography():
    assert normalize_text("Trov&#230;rdig og p&#229;lidelig") == "Troværdig og pålidelig"
    assert normalize_text("don&#39;t  stop &#128077;") == "don't stop 👍"
    assert normalize_text("don’t go…") == "don't go..."
    assert normalize_text("don´t") == "don't"
    assert normalize_text("a​b   c\n\n\nd") == "ab c\nd"


def test_brand_variants_map_to_one_spelling():
    for variant in ["trust pilot", "Trust-Pilot", "TRUSTPILOT's", "Truspilot", "trustpliot", "turstpilot",
                    "Trustpilot.com", "TP"]:
        assert canonicalize_brand(f"I like {variant} a lot").lower() == "i like trustpilot a lot", variant


def test_brand_canonicalization_leaves_ordinary_words_alone():
    text = "I trust this site, trusted and trustworthy. Testing tp lowercase."
    assert canonicalize_brand(text) == text


def test_merge_title_drops_redundant_titles():
    body = "Trustpilot is an excellent platform for our customers"
    assert merge_title("Trustpilot is an excellent platform for...", body) == (body, False)  # auto-truncated
    assert merge_title(body, body) == (body, False)  # identical
    assert merge_title("excellent platform", body) == (body, False)  # no new words
    assert merge_title("Review deleted", body) == ("Review deleted\n" + body, True)  # adds words


def test_truncate_text_cuts_at_word_boundary():
    assert truncate_text("short review", 100) == "short review"
    out = truncate_text("one two three four five", 12)
    assert out == "one two [...]" and len(out) <= 12 + len(" [...]")


def test_low_content():
    assert is_low_content("👍👍👍")
    assert is_low_content("Excellent service!")
    assert not is_low_content("My genuine review was removed without reason")
