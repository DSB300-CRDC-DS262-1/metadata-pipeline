# DSB300 Capstone Project
## Cotton Research & Development Corporation (CRDC): AI Metadata Pipeline (text-based extraction)
This repository is for the development of a python application that can extract structured, reliable quantitative findings from unstructured research documents.

![Tests](https://github.com/DSB300-CRDC-DS262-1/metadata-pipeline/actions/workflows/tests.yml/badge.svg)

Extracts metadata and prose-reported quantitative statistics from CRDC
research PDFs into structured JSON (`documents` / `contexts` /
`statistics`, per `data/schema_worked_examples.json`). Text-layer
statistics only: tables and images are a separate team's scope.

Status: ingestion (layout detection, symbol repair, image fallback) and
document metadata extraction are built and tested, running on local models
by default. Statistics extraction is on `main`; contexts are not built yet.

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) for local models.

```bash
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
ollama pull gemma4:e4b    # metadata and page images (default)
```

Gemini is optional. To use it, copy `.env.example` to `.env` and set
`GEMINI_API_KEY`. `.env` is gitignored, so never commit a real key.

## Adding the sample PDFs

1. Copy the four source PDFs into `data/raw_pdfs/` (gitignored: they are
   copyrighted third-party papers). `data/document_manifest.json` lists each
   document, its filename and where it was downloaded from (`retrieved_from`).
2. Run `pytest`; `test_sample_documents.py` picks them up.

## Running the tests

```bash
pytest
```

Synthetic-PDF and mocked-model tests need nothing else. Tests on the real
PDFs and the live Gemini test skip when the PDFs or API key are missing, so
CI (`.github/workflows/tests.yml`, on every PR into `main`) stays free.

## Metadata schema

`DocumentMetadata` in `extraction/metadata.py` describes the document only:

| Field | Meaning |
|---|---|
| `title`, `document_type`, `authors` | As printed. A corporate author (e.g. "Voconiq (2025)") is listed as the author |
| `organisation` | Who did the work: every institution the authors belong to, at the top level |
| `publisher`, `funder`, `copyright_holder` | Who released it, who paid for or commissioned it, who holds copyright |
| `publication_date` | When first made public (a journal article's "Available online" date), ISO 8601 |
| `journal_article` | Journal, volume, issue year, pages, ISSN and received / revised / accepted dates. Null for anything that is not a journal article |
| `doi` | The document's own DOI, without the `https://doi.org/` prefix |
| `keywords` | Only when the document prints a keyword list |
| `series`, `supersedes_note` | The report series, and the sentence saying it updates an earlier report |
| `contact_emails` | Email addresses printed in the document |
| `evidence` | For every filled field: page number, exact quote, and whether it is stated or inferred |

How the data behind a statistic was produced (time period, region,
population, data source, collection method, currency basis, limitations) is
recorded in `contexts`, not here. A document has one or more contexts, each
shared by the statistics measured under the same conditions.

In the reference file a field can list several acceptable answers as
`{"accept": [...]}` where the PDF supports more than one reading.

## How metadata is extracted

1. **Text**: pdfplumber, with two-column pages read column by column.
2. **What the model sees**: pages 1, 2 and the last page in full, plus lines
   from other pages that mention funding, acknowledgements, copyright or
   updates. Reference lists are left out (citations carry other documents'
   metadata). Pages are marked `=== page N ===` for the evidence list.
3. **Model**: returns JSON matching the schema; list lengths are capped so a
   model cannot loop past its context, and a cut-off answer is retried once.
4. **Plain-code rules** (`tidy()` in `metadata.py`), applied after the model:
   - Details printed in a fixed layout override the model
     (`extraction/printed.py`): article history dates, the DOI on the first
     two pages, and the journal header with an ISSN check-digit test.
   - Dates become ISO 8601; words like "null" or "n/a" become empty.
   - Organisation names are standardised with `data/organisation_names.json`
     (e.g. "Australian Cotton Research and Development Corporation (CRDC)"
     becomes "Cotton Research and Development Corporation").
   - If the model's own funding or copyright quote names more organisations
     than its answer, they are added (models often stop after the first).
   - Keywords are dropped unless a keyword list is printed; a corporate
     author fills an empty organisation.

Email addresses are kept; set `REDACT_EMAILS = True` in `loader.py` to strip
them before any text reaches a model.

## Running metadata extraction

```bash
.venv/Scripts/python.exe scripts/run_metadata_extraction.py                              # Gemma 4 E4B, local
.venv/Scripts/python.exe scripts/run_metadata_extraction.py --model gemma4:12b --think off
.venv/Scripts/python.exe scripts/run_metadata_extraction.py --backend gemini             # needs a key
```

Results go to `output/metadata/<backend>_<model>/<document_id>.json` with
time and token counts, and each field is printed against the reference.

| Option | Use |
|---|---|
| `--model` | Any Ollama model, or a Gemini model with `--backend gemini` |
| `--think off` | Turn off the reasoning step. Required for `gemma4:12b` and `gemma4:26b`, which otherwise reason until the context is full |
| `--num-ctx` | Context window (default 32768). 8192 is enough for metadata and saves GPU memory |
| `--num-gpu` | How many model layers go on the GPU; the rest run on the CPU (see below) |
| `--seed`, `--suffix` | Fixed sampling seed; suffix for repeat runs (`r2`, `r3`) |

## Choosing a local model

Results on the current schema (76 checks across the four documents):

| Model | Score | Seconds per doc | GPU memory | Notes |
|---|---|---|---|---|
| Gemma 4 12B, `--think off` | 75-76 | 7 | 9.5 GB | Fastest on a 16 GB GPU; sometimes misses Voconiq's copyright holder |
| Phi-4 | 75 | 10 | 11.7 GB | Same score every run; leaves Hedayati's organisation empty |
| Gemma 4 E4B (default) | 76 | 8 | 5.3 GB | Smallest; fits 4-8 GB GPUs; 76 on every fresh run since the latest rules |
| Gemma 4 26B (A4B), `--think off` | 72 | 9 | 14.3 GB | Does not fit a 16 GB GPU fully |

Measured on an RTX 5070 Ti (16 GB). Energy and memory can be re-measured
with `scripts/measure_compute.py`.

**Smaller GPUs.** Ollama under-reports Gemma 4's memory needs, so set
`--num-gpu` yourself. Simulated on the same machine with a fast CPU (8K
context), so expect laptops to be slower:

| GPU memory | Best choice | Layers on GPU (`--num-gpu`) | Score | Seconds per doc |
|---|---|---|---|---|
| 8 GB | `gemma4:e4b` | 42 (all) | 76 | 10 |
| 4 GB | `gemma4:e4b` | 40 | 76 | 12 |
| 2 GB | `gemma4:12b --think off` | 8 | 74 | 52 |

The first call loads the model (20-80 seconds). Earlier comparison runs are
kept under `output/metadata_v*`.

## Pages the text layer cannot read

Some PDFs draw characters whose fonts carry no text (Voconiq prints "73%",
the text layer reads "3%"). The loader counts these per page
(`Page.unreadable_glyphs`) and warns.

Some fonts label a symbol with the wrong character (Hedayati's "974e985" is
really "974–985"). The loader flags a single character set in a different font
from its neighbours, reads an image of that word with a vision model
(`load_document(..., read_text=...)`), and once two reads agree saves a rule
to `data/symbol_rules.json`. Saved rules apply on every load with no model.
Joined letters (ﬁ, ﬂ) are split back into plain letters.

Flagged or empty pages can be re-read from the page image:

```bash
.venv/Scripts/python.exe scripts/transcribe_unreadable_pages.py
```

Only running text is transcribed; charts and tables become `[figure]`
because they belong to the table/image workstream. Gemma 4 E4B reads a page in
about 3 seconds (12B takes 7) with the same accuracy; output goes to `output/vision/`.

## Comparing and measuring

```bash
.venv/Scripts/python.exe scripts/compare_metadata_runs.py              # runs side by side, repeat-run spread, majority vote
.venv/Scripts/python.exe scripts/measure_compute.py gemma4:e4b phi4      # GPU memory, power, energy per document
.venv/Scripts/python.exe scripts/compare_vision_models.py gemma4:e4b   # page and word reading
.venv/Scripts/python.exe scripts/verify_ground_truth.py                # reference statistics against the PDFs
```

## Layout

```
src/crdc_pipeline/
  ingestion/
    layout.py    column detection (pure function, no PDF library dependency)
    loader.py    PDF -> Document/Page; text, symbol repair and image fallback
    symbols.py   finds wrongly labelled and hidden characters, learns font rules
  extraction/
    backends.py  Gemini and Ollama backends, one interface
    metadata.py  metadata schema, prompt, extraction and tidy rules
    printed.py   fixed-layout details (article history, DOI, journal header)
    names.py     standard organisation names
    scoring.py   field matching against the reference
    vision.py    page and word reading from images
scripts/
  run_metadata_extraction.py      runs metadata extraction across the manifest
  compare_metadata_runs.py        compares saved runs against the reference
  measure_compute.py              GPU memory, power and energy per model
  transcribe_unreadable_pages.py  re-reads flagged pages with a vision model
  compare_vision_models.py        compares vision models on page and word reading
  verify_ground_truth.py          checks reference statistics against the PDFs
tests/
  test_layout.py, test_loader.py   column detection and loading (synthetic PDFs)
  test_symbols.py                  wrongly labelled symbol detection and rule learning
  test_metadata_extraction.py      schema, prompt and tidy rules
  test_printed.py                  article history, DOI, journal header, ISSN
  test_backends.py                 backends with mocked HTTP (no server needed)
  test_scoring.py                  field matching
  test_sample_documents.py         checks against the real sample PDFs
data/
  raw_pdfs/                        the four source PDFs (gitignored)
  document_manifest.json           document_id, filename and download link
  schema_worked_examples.json      reference documents, contexts and statistics
  ground_truth_all_documents.json  48-statistic reference set for recall
  organisation_names.json          standard name and known variants per organisation
  symbol_rules.json                learned corrections for wrongly labelled symbols
docs/
  weaknesses_and_fixes.md          known extraction problems and planned fixes
  architecture.md                  design notes for the ingestion module
```
