# DSB300 Capstone Project
## Cotton Research & Development Corporation (CRDC): AI Metadata Pipeline (text-based extraction)
This repository is for the development of a python application that can extract structured, reliable quantitative findings from unstructured research documents.

![Auto Assign](https://github.com/DSB300-CRDC-DS262-1/metadata-pipeline/actions/workflows/auto-assign.yml/badge.svg)

Extracts metadata and prose-reported quantitative statistics from CRDC
research PDFs into structured JSON (`documents` / `contexts` /
`statistics`, per `data/schema_worked_examples.json`). Text-layer
statistics only -- tables and images are a separate team's scope.

Status: ingestion and layout detection built and tested. LLM extraction
not started yet.

## Setup

Requires Python 3.11+ and the poppler `pdftotext` binary on PATH (used
for single-column page extraction; two-column pages are handled with
pdfplumber alone).

```bash
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
```

## Running the tests

```bash
pytest
```

`tests/test_layout.py` and `tests/test_loader.py` run against synthetic
PDFs and need nothing else. `tests/test_sample_documents.py` checks the
real CRDC sample PDFs once they're added -- see below.

## Adding the sample PDFs

1. Copy the four source PDFs into `data/raw_pdfs/` (gitignored --
   these are copyrighted third-party research papers, not ours to
   redistribute; see `data/document_manifest.json` for the four
   documents and their sources).
2. Fill in the `filename` field for each entry in
   `data/document_manifest.json`.
3. Re-run `pytest` -- `test_sample_documents.py` will pick them up.

## Layout

```
src/crdc_pipeline/
  ingestion/
    layout.py   column detection (pure function, no PDF library dependency)
    loader.py   PDF -> Document/Page, routes each page through the
                extraction method its detected layout needs
tests/
  test_layout.py           unit tests, synthetic word positions
  test_loader.py            integration tests, synthetic PDFs (reportlab)
  test_sample_documents.py  sanity check against the real sample PDFs
data/
  raw_pdfs/                  the four source PDFs go here (gitignored)
  document_manifest.json     document_id -> filename mapping
  ground_truth_all_documents.json   48-statistic reference set for recall
  schema_worked_examples.json       target output schema
docs/
  weaknesses_and_fixes.md    known extraction problems and planned fixes
  architecture.md            design notes for the ingestion module
output/                       extraction output lands here (not yet used)
```

See `docs/architecture.md` for why column detection works the way it
does, and `docs/weaknesses_and_fixes.md` for the extraction problems
this and later modules are designed around.
