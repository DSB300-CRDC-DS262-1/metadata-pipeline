# Ingestion & layout detection: design notes

Covers `src/crdc_pipeline/ingestion/`. See `weaknesses_and_fixes.md` for
the underlying problems this module addresses.

## Why detect columns before extracting text at all

Issue 1 in `weaknesses_and_fixes.md`: `pdftotext -layout` reads a
two-column page left-to-right, line by line, so it splices unrelated
sentences from each column onto the same output line. No amount of
prompting downstream can recover the correct reading order once it's
been scrambled, so this has to be fixed at ingestion, before any text
reaches an LLM.

## Detection approach

`layout.detect_columns()` takes a list of word bounding boxes (from
pdfplumber's `extract_words()`) and page width, with no PDF library
dependency of its own -- it's a pure function over geometry, which is
what makes it unit-testable without any real PDF file (see
`tests/test_layout.py`).

Two passes:

1. **Find a candidate gutter.** Bin all word positions across the page
   width and look for the widest empty band, excluding a margin at each
   edge. A genuine column gutter is typically 15-30pt wide on an
   otherwise fully-set page; the margin exclusion and minimum-width
   threshold rule out incidental whitespace near the page edges.

2. **Validate it against actual lines.** A global gap in the word
   histogram can also come from a full-width heading, abstract, or
   figure sitting above/below a two-column body -- exactly the case in
   the Hedayati paper, where the ground truth cites its abstract
   ("page: abstract") separately from the two-column body pages. So the
   candidate gutter only counts if a majority of the page's actual text
   *lines* (grouped by vertical position) don't cross it. A full-width
   heading is one line out of many and doesn't change the majority; a
   page that's mostly full-width text with an incidental gap fails this
   check and correctly comes back single-column.

This runs **per page**, not per document, because column layout can
vary within one document (title/abstract vs. two-column body). The
document-level `detected_layout` field (`single_column` / `two_column`
/ `mixed`) is a summary for logging and the manifest sanity check, not
what extraction routing actually uses.

### Two things the real Hedayati PDF surfaced that a synthetic test wouldn't

Testing against the actual sample PDFs (not just synthetic reportlab
fixtures) surfaced two problems no synthetic test had anticipated:

1. **The real gutter is narrow -- about 3% of page width (~18pt on a
   595pt-wide A4 page)**, not the wider magazine-style gap a first
   guess might assume. `min_gap_ratio` is tuned to 0.02 (2%) with
   400 histogram bins so the search has enough resolution not to lose
   an 18pt gap to bin-quantization error.

2. **A running header ("M. Hedayati et al. / Journal of Cleaner
   Production...") is extracted as a single word spanning the full
   page width**, because this PDF's font encoding doesn't reliably
   mark word-breaks (the same underlying issue as the missing spaces
   below). One such word crossing straight over the true gutter is
   enough to hide it from a page-wide search, since the search requires
   a totally empty band. Fix: exclude a vertical band at the top and
   bottom of the page (`vertical_margin_ratio`, default 8%) from the
   gutter search itself, while still including those lines in the
   per-line validation, where one outlier line barely moves the
   confidence score.

### Why pdfplumber alone, not pdftotext for the single-column path

An earlier version of this module used `pdftotext -layout` for
single-column pages and pdfplumber only for the two-column crop. That
turned out to be an unexamined default, not a technical requirement:
pdfplumber's own `extract_text(layout=True)` reconstructs whitespace
the same way, and testing it against `pdftotext -layout` on real
single-column pages (including a stats-heavy table page) showed
matching content -- if anything, pdfplumber preserved table row
alignment slightly better on one page. Dropping `pdftotext` removes an
external binary dependency and the fragility that comes with it: the
`pdftotext` bundled with Git for Windows is Xpdf-derived and doesn't
support the same flags as a poppler build would (this bit us once
already, when a crop-box approach was briefly considered), so
correctness would otherwise depend on which `pdftotext` happened to be
on a given machine's PATH. Everything now goes through pdfplumber.

### pdfplumber's word-spacing on tightly-kerned text

Extraction initially reproduced text like
`agriculturalresources(Kongetal.,2015)` on the Hedayati PDF -- missing
spaces between words, confirmed (during the `pdftotext` comparison
above) to be a pdfplumber-specific issue: plain `pdftotext` renders the
same page as `agricultural resources (Kong et al., 2015)`. This is
pdfplumber's own word-boundary heuristic (`x_tolerance`, default 3pt)
being too loose for this PDF's character spacing, not a text-layer gap
-- the characters are all present, pdfplumber just wasn't inserting a
space between them. Lowering `x_tolerance` to 1.5 (`loader.X_TOLERANCE`,
applied to both extraction paths) matches plain pdftotext's output.

### Unicode dash normalisation

PDF fonts often use a typographic hyphen/dash glyph distinct from a
plain ASCII `-` (confirmed on aghealth2019: "3‐5 years" uses HYPHEN
U+2010). Left alone, every later regex that looks for a number range or
a hyphenated compound would need to account for each dash variant
separately. Since this pipeline only cares about the underlying value,
not exact source typography, `loader._normalise()` maps every
dash-like glyph (hyphen, non-breaking hyphen, figure dash, en dash, em
dash, minus sign) to ASCII `-` once, at ingestion, rather than leaving
that to every consumer downstream.

## Extraction routing

- **Single-column pages**: `pdfplumber.extract_text(layout=True)`.
  Position-preserving mode is exactly what you want when there's only
  one column -- it keeps paragraph structure and avoids re-flowing text
  across line breaks awkwardly.
- **Two-column pages**: the same library, cropped at the detected
  gutter x into left and right halves, each extracted independently and
  concatenated left-then-right. This reconstructs reading order
  directly from geometry rather than trusting a layout-preserving
  extractor's line-by-line assumptions, which is exactly what breaks on
  a two-column page (issue 1).

## What this module deliberately does not do yet

- **Text-layer gaps (issue 2)**: a page that produces zero characters
  gets a warning (`Page.warnings`), not a silent empty string, and not
  an attempted fix. Recovering those values needs OCR or a multimodal
  model reading the rendered page image -- a separate fallback stage,
  scoped but not built here.
- **`value_status` / `record_type` / cross-referencing / context
  deduplication** (issues 3-8): these are extraction-stage and schema
  concerns, not ingestion. This module's job ends at handing clean,
  correctly-ordered per-page text (plus provenance: which method
  extracted it, and any warnings) to whatever reads it next.
