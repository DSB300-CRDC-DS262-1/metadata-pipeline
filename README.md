# DSB300 Capstone Project
## Cotton Research & Development Corporation (CRDC): AI Metadata Pipeline (text-based extraction)
This repository is for the development of a python application that can extract structured, reliable quantitative findings from unstructured research documents.

![Tests](https://github.com/DSB300-CRDC-DS262-1/metadata-pipeline/actions/workflows/tests.yml/badge.svg)

Extracts metadata and prose-reported quantitative statistics from CRDC
research PDFs into structured JSON (`documents` / `contexts` /
`statistics`, per `data/schema_worked_examples.json`). Text-layer
statistics only: tables and images are a separate team's scope.

Status: ingestion (layout detection, symbol repair, image fallback),
document metadata, and statistics with their contexts are built and tested,
running on local models (Gemma 4 E4B) by default.

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

1. Download the source PDFs into `data/raw_pdfs/` (gitignored: they are
   copyrighted third-party papers). `data/document_manifest.json` lists the
   four documents CRDC provided, their filenames and where each was
   downloaded from (`retrieved_from`).
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
   - If the model's own funding, copyright or collaboration quote names more
     organisations than its answer, they are added (models often stop after
     the first). For journal articles, every university or department in the
     affiliation list is added.
   - The document's own suggested citation ("Citation: Roth Rural (2017) ...")
     settles the year, publisher, full title, and a corporate author.
   - Emails and a "Published: October 2024" date are read from the whole
     document; a later copyright year sets the publication year.
   - Evidence is checked: funders backed only by a figure's source line, a
     copyright line, an inferred quote, or a sentence where they only received
     money are dropped; a copyright holder needs a printed copyright line;
     someone who only managed the project is not an author.
   - Keywords are dropped unless a keyword list is printed; a corporate
     author fills an empty organisation; document types are kept to the
     allowed list ("survey" in the title gives a survey report).
   - Long organisation names damaged by OCR take the closest known name.

Email addresses are kept; set `REDACT_EMAILS = True` in `loader.py` to strip
them before any text reaches a model.

## Running metadata extraction

```bash
.venv/Scripts/python.exe scripts/run_metadata_extraction.py                              # Gemma 4 E4B, local
.venv/Scripts/python.exe scripts/run_metadata_extraction.py --model gemma4:12b --think off
.venv/Scripts/python.exe scripts/run_metadata_extraction.py --backend gemini             # needs a key
.venv/Scripts/python.exe scripts/run_metadata_extraction.py --vision                     # read scanned pages from the image
```

Results go to `output/metadata/<backend>_<model>/<document_id>.json` with
time and token counts, and each field is printed against the reference.

With `--vision`, Gemma 4 E4B gets all 76 reference fields right across the
four documents (19 fields each), in about 10 seconds per document.

| Option | Use |
|---|---|
| `--model` | Any Ollama model, or a Gemini model with `--backend gemini` |
| `--think off` | Turn off the reasoning step. Required for `gemma4:12b` and `gemma4:26b`, which otherwise reason until the context is full |
| `--num-ctx` | Context window (default 32768). 8192 is enough for metadata and saves GPU memory |
| `--num-gpu` | How many model layers go on the GPU; the rest run on the CPU (see below) |
| `--seed`, `--suffix` | Fixed sampling seed; suffix for repeat runs (`r2`, `r3`) |
| `--vision` | Read pages whose text is garbled (poor OCR of a scan), empty, or has OCR-split email addresses from the page image |
| `--documents` | Only these document IDs |

## Choosing a local model

Results from the earlier comparison (76 checks across the first four
documents):

| Model | Score | Seconds per doc | GPU memory | Notes |
|---|---|---|---|---|
| Gemma 4 12B, `--think off` | 75-76 | 7 | 9.5 GB | Fastest on a 16 GB GPU; sometimes misses Voconiq's copyright holder |
| Phi-4 | 75 | 10 | 11.7 GB | Same score every run; leaves Hedayati's organisation empty |
| Gemma 4 E4B (default) | 76 | 8 | 5.3 GB | Smallest; fits 4-8 GB GPUs; 76 on every fresh run since the latest rules |
| Gemma 4 26B (A4B), `--think off` | 72 | 9 | 14.3 GB | Does not fit a 16 GB GPU fully |

Measured on an RTX 5070 Ti (16 GB).

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

## Statistics and contexts

`extraction/statistics.py` reads a document page by page, so a small local
model never needs the whole document at once:

1. **Context items** per page: data source, collection method, population,
   region, time period, inclusion and exclusion, method, assumptions and
   limitations, functional unit, currency year, survey questions, scenarios.
2. **Whole-study scope**: one call picks the best item for each key kind
   (period, region, sample, source, method); a second, narrower call picks
   method facts that apply to all results. These reach every statistic the
   study reports itself, but not figures cited from other studies.
3. **Statistics** per page, then a **linking** call per page for the
   specific items each number needs (its survey question, scenario, source).
4. **Plain-code rules**: the reading the wording states ("just over" is a
   lower bound), values as printed ("80%" is 80, "half" is 50%), the period
   nearest the value in its sentence at printed precision, survey answers
   dated to the fieldwork year, citations make a figure "cited". Table rows,
   chart values, reference lists, date-only values and numbers that only set
   up the report (subgroup definitions, scale ends, "three main themes") are
   dropped; a bracketed unit conversion and the two ends of a printed range
   are kept as one statistic.
5. **Duplicates**: numbers with the same value and unit are paired, and one
   model call per document judges which pairs are the same measurement (the
   abstract repeating a result). The later one is dropped and the earlier
   keeps both sets of links; different survey questions or groups are kept
   apart. `--no-dedupe` turns this off.

```bash
.venv/Scripts/python.exe scripts/run_statistics_extraction.py --vision   # reference documents, local E4B
.venv/Scripts/python.exe scripts/run_statistics_extraction.py --score-only output/statistics/<run>
```

Scored against `data/findings_reference.json` (four documents, 238
statistics), the latest E4B run with `--vision` finds 98% of the reference
statistics (233) at 88% precision. Field agreement on matched statistics:
value reading 99%, type 92%, period 91%, record type 85% (repeat runs vary by
a few points). Context links reach 74% of the facts the reference links, with
35% precision. Key conditions are right for nearly all statistics: region 200
of 203, time period 161 of 162, sample 106 of 113, data source 93 of 110. A
run takes about 2 to 7 minutes per document.

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

## Checking the reference

```bash
.venv/Scripts/python.exe scripts/verify_ground_truth.py   # reference statistics against the PDFs
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
    printed.py   fixed-layout details (article history, DOI, journal header, citation, emails)
    names.py     standard organisation names
    scoring.py   field matching against the reference
    findings.py  statistics, context items and contexts schema
    statistics.py            statistics and context extraction, step by step
    statistics_prompts.py    prompts and response schemas for each step
    statistics_filters.py    what to leave out: reference lists, table rows, set-up numbers
    statistics_values.py     each value's reading, bounds, type and record type
    statistics_periods.py    periods at the precision printed
    statistics_document.py   study scope, duplicates, links and record types across a document
    findings_scoring.py   statistics and context links against the reference
    vision.py    page and word reading from images
scripts/
  run_metadata_extraction.py      runs metadata extraction across the manifest
  run_statistics_extraction.py    runs statistics and context extraction and scores it
  transcribe_unreadable_pages.py  re-reads flagged pages with a vision model
  verify_ground_truth.py          checks reference statistics against the PDFs
tests/
  test_layout.py, test_loader.py   column detection and loading (synthetic PDFs)
  test_symbols.py                  wrongly labelled symbol detection and rule learning
  test_metadata_extraction.py      schema, prompt and tidy rules
  test_printed.py                  article history, DOI, journal header, ISSN
  test_backends.py                 backends with mocked HTTP (no server needed)
  test_scoring.py                  field matching
  test_findings.py                 statistics and context schema, grouping into contexts
  test_statistics_extraction.py    statistics pipeline with a fake model, and its rules
  test_findings_scoring.py         statistics and context link scoring
  test_findings_reference.py       reference statistics fit the schema and are quoted from their page
  test_sample_documents.py         checks against the real sample PDFs
data/
  raw_pdfs/                        the four source PDFs (gitignored)
  document_manifest.json           document_id, filename and download link
  schema_worked_examples.json      metadata reference for every document
  findings_reference.json          reference statistics, context items and contexts (four documents)
  ground_truth_all_documents.json  48-statistic reference set for recall
  organisation_names.json          standard name and known variants per organisation
  symbol_rules.json                learned corrections for wrongly labelled symbols
docs/
  weaknesses_and_fixes.md          known extraction problems and planned fixes
  GitHelper.md                     git workflow for the team
  architecture.md                  design notes for the ingestion module
```
