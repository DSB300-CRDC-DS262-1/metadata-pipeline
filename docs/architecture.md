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

## Extraction routing

- **Single-column pages**: `pdftotext -layout`. Position-preserving
  mode is exactly what you want when there's only one column -- it
  keeps paragraph structure and avoids re-flowing text across line
  breaks awkwardly.
- **Two-column pages**: pdfplumber, cropped at the detected gutter x
  into left and right halves, each extracted independently and
  concatenated left-then-right. This reconstructs reading order
  directly from geometry rather than trusting either tool's stream
  order -- plain `pdftotext` (the naive fix for issue 1) happens to
  work when the PDF's content stream was written in column order, but
  that's a property of how the PDF was generated, not something this
  pipeline can assume for every future document.

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
