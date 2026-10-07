"""Mislabelled-glyph detection and rule learning, on synthetic character lists."""

from crdc_pipeline.ingestion.symbols import SymbolRules


def chars_for(text: str, odd: dict[int, str]) -> list[dict]:
    """One glyph per character, touching, on one line; positions in `odd` use another font."""
    chars = []
    for i, ch in enumerate(text):
        font = odd.get(i, "ABC+Body")
        chars.append({"text": ch, "fontname": font, "x0": i * 5.0, "x1": i * 5.0 + 5.0,
                      "top": 100.0, "bottom": 110.0})
    return chars


def test_flags_single_glyph_in_another_font_inside_a_word() -> None:
    chars = chars_for("974e985", {3: "XYZ+AdvSymbol"})
    assert SymbolRules().suspicious(chars) == [3]


def test_ignores_whole_words_in_another_font_and_ligatures() -> None:
    italic = chars_for("abc", {0: "ABC+Italic", 1: "ABC+Italic", 2: "ABC+Italic"})
    ligature = chars_for("ofl", {1: "ABC+Ligatures"})
    ligature[1]["text"] = "ﬁ"
    assert SymbolRules().suspicious(italic) == []
    assert SymbolRules().suspicious(ligature) == []


def test_learns_rule_after_two_agreeing_reads_and_saves_it(tmp_path) -> None:
    path = tmp_path / "rules.json"
    rules = SymbolRules(path)
    chars = chars_for("974e985 12e34", {3: "XYZ+AdvSymbol", 10: "XYZ+AdvSymbol"})
    chars[7]["text"] = " "
    reads = iter(["974–985", "12–34"])

    rules.learn(chars, crop=lambda bbox: b"png", read_text=lambda png: next(reads), source="doc")
    rules.apply(chars)

    assert chars[3]["text"] == "–" and chars[10]["text"] == "–"
    assert chars[3]["original_text"] == "e"
    assert SymbolRules(path).replacement({"text": "e", "fontname": "QQQ+AdvSymbol"}) == "–"


def test_rejects_reads_that_disagree_elsewhere_in_the_word() -> None:
    rules = SymbolRules()
    chars = chars_for("974e985", {3: "XYZ+AdvSymbol"})
    rules.learn(chars, crop=lambda bbox: b"png", read_text=lambda png: "971-985", source="doc")
    rules.apply(chars)
    assert chars[3]["text"] == "e"
