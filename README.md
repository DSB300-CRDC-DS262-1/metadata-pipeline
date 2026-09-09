# DSB300 Capstone Project
## Cotton Research & Development Corporation (CRDC): AI Metadata Pipeline (text-based extraction)
This repository is for the development of a python application that can extract structured, reliable quantitative findings from unstructured research documents.

![Tests](https://github.com/DSB300-CRDC-DS262-1/metadata-pipeline/actions/workflows/tests.yml/badge.svg)

Extracts metadata and prose-reported quantitative statistics from CRDC
research PDFs into structured JSON (`documents` / `contexts` /
`statistics`, per `data/schema_worked_examples.json`). Text-layer
statistics only -- tables and images are a separate team's scope.

Status: ingestion and layout detection built and tested. Metadata
extraction (title/authors/organisation/date/URL) built, via a frontier
model (Gemini). Statistics extraction not started yet.

## Setup

Requires Python 3.11+. Pure Python for ingestion -- no external PDF
binary needed, all extraction goes through pdfplumber.

```bash
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
```

Metadata extraction calls the Gemini API and needs a key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your-key-here
```

`.env` is gitignored -- never commit a real key.

## Running the tests

```bash
pytest
```

`tests/test_layout.py` and `tests/test_loader.py` run against synthetic
PDFs and need nothing else. `tests/test_sample_documents.py` checks the
real CRDC sample PDFs once they're added -- see below.

Runs automatically on every PR into `main` (`.github/workflows/tests.yml`).
The real-sample-PDF and live-Gemini tests skip in CI the same way they
skip locally without the PDFs/API key -- CI stays deterministic and
free rather than burning API quota on every push.

## Adding the sample PDFs

1. Copy the four source PDFs into `data/raw_pdfs/` (gitignored --
   these are copyrighted third-party research papers, not ours to
   redistribute; see `data/document_manifest.json` for the four
   documents and their sources).
2. Fill in the `filename` field for each entry in
   `data/document_manifest.json`.
3. Re-run `pytest` -- `test_sample_documents.py` will pick them up.

## Running metadata extraction

```bash
.venv/Scripts/python.exe scripts/run_metadata_extraction.py
```

Runs all four manifest documents through Gemini, writes each result to
`output/metadata/<document_id>.json`, and prints a field-by-field
comparison against the reference metadata in
`data/schema_worked_examples.json`.

## Layout

```
src/crdc_pipeline/
  ingestion/
    layout.py    column detection (pure function, no PDF library dependency)
    loader.py    PDF -> Document/Page, routes each page through the
                 extraction method its detected layout needs
  extraction/
    metadata.py  frontier-model (Gemini) document metadata extraction
scripts/
  run_metadata_extraction.py  runs metadata extraction across the manifest
tests/
  test_layout.py              unit tests, synthetic word positions
  test_loader.py               integration tests, synthetic PDFs (reportlab)
  test_sample_documents.py     sanity check against the real sample PDFs
  test_metadata_extraction.py  prompt/page-selection tests; live API test
                                skips unless GEMINI_API_KEY is set
data/
  raw_pdfs/                  the four source PDFs go here (gitignored)
  document_manifest.json     document_id -> filename mapping
  ground_truth_all_documents.json   48-statistic reference set for recall
  schema_worked_examples.json       target output schema (incl. reference
                                     metadata for comparison)
docs/
  weaknesses_and_fixes.md    known extraction problems and planned fixes
  architecture.md            design notes for the ingestion module
output/
  metadata/                  per-document metadata extraction results
```

See `docs/architecture.md` for why column detection works the way it
does, and `docs/weaknesses_and_fixes.md` for the extraction problems
this and later modules are designed around.
