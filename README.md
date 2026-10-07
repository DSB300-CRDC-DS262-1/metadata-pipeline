# DSB300 Capstone Project
## Cotton Research & Development Corporation (CRDC): AI Metadata Pipeline (text-based extraction)
This repository is for the development of a python application that can extract structured, reliable quantitative findings from unstructured research documents.

![Tests](https://github.com/DSB300-CRDC-DS262-1/metadata-pipeline/actions/workflows/tests.yml/badge.svg)

> **About this branch.** `amaan/full-local-extraction` is a learning branch and will not be merged into
> `main`. It runs the whole pipeline (metadata, statistics and their contexts) on a local model, Gemma 4 E4B
> through Ollama, with plain-code rules that check the model's answers, and scores every stage against
> references checked by hand against the four PDFs CRDC provided.
>
> **How it differs from `main`:** `main` has the same local metadata extraction (PR #9) in a flat layout, and
> statistics and context extraction through Gemini. Here the code is organised by extraction type
> (`extraction/metadata/`, `contexts/`, `statistics/`), statistics and contexts are extracted page by page by
> the local model in one pass, and the references are in `data/reference/` (metadata, statistics, contexts).
>
> **Results on the four documents:** metadata 76 of 76 fields; statistics recall 99% (237 of 239) and
> precision 96%; context links recall 85% and precision 57% against the reference.
>
> **Where to look:** the rules are in `src/crdc_pipeline/extraction/*/` and each one has a test in `tests/`;
> "How metadata is extracted" and "Statistics and contexts" below explain them in order.

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
2. Run `pytest`; `tests/ingestion/test_sample_documents.py` picks them up.

## Running the tests

```bash
pytest
```

Synthetic-PDF and mocked-model tests need nothing else. Tests on the real
PDFs and the live Gemini test skip when the PDFs or API key are missing, so
CI (`.github/workflows/tests.yml`, on every PR into `main`) stays free.

## Metadata schema

`DocumentMetadata` in `extraction/metadata/extract.py` describes the document only:

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

In the metadata reference (`data/reference/metadata.json`) a field can list
several acceptable answers as `{"accept": [...]}` where the PDF supports more than one reading.

## How metadata is extracted

1. **Text**: pdfplumber, with two-column pages read column by column.
2. **What the model sees**: pages 1, 2 and the last page in full, plus lines
   from other pages that mention funding, acknowledgements, copyright or
   updates. Reference lists are left out (citations carry other documents'
   metadata). Pages are marked `=== page N ===` for the evidence list.
3. **Model**: returns JSON matching the schema; list lengths are capped so a
   model cannot loop past its context, and a cut-off answer is retried once.
4. **Plain-code rules** (`tidy()` in `metadata/extract.py`), applied after the model:
   - Details printed in a fixed layout override the model
     (`extraction/metadata/printed.py`): article history dates, the DOI on the first
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
.venv/Scripts/python.exe scripts/metadata/run_metadata_extraction.py                              # Gemma 4 E4B, local
.venv/Scripts/python.exe scripts/metadata/run_metadata_extraction.py --model gemma4:12b --think off
.venv/Scripts/python.exe scripts/metadata/run_metadata_extraction.py --backend gemini             # needs a key
.venv/Scripts/python.exe scripts/metadata/run_metadata_extraction.py --no-vision                  # keep unreadable text layers as they are
```

Results go to `output/metadata/<backend>_<model>/<document_id>.json` with
time and token counts, and each field is printed against the reference.

Gemma 4 E4B gets all 76 reference fields right across the
four documents (19 fields each), in about 10 seconds per document.

| Option | Use |
|---|---|
| `--model` | Any Ollama model, or a Gemini model with `--backend gemini` |
| `--think off` | Turn off the reasoning step. Required for `gemma4:12b` and `gemma4:26b`, which otherwise reason until the context is full |
| `--num-ctx` | Context window (default 32768). 8192 is enough for metadata and saves GPU memory |
| `--num-gpu` | How many model layers go on the GPU; the rest run on the CPU (see below) |
| `--seed`, `--suffix` | Fixed sampling seed; suffix for repeat runs (`r2`, `r3`) |
| `--no-vision` | Keep the text layer as it is. By default, pages whose text is garbled (poor OCR of a scan), empty, or has OCR-split email addresses are read from the page image |
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

`extraction/statistics/extract.py` reads a document page by page, so a small local
model never needs the whole document at once:

1. **Context items** per page: data source, collection method, population,
   region, time period, inclusion and exclusion, method, assumptions and
   limitations, functional unit, currency year, survey questions, scenarios.
2. **Whole-study scope**: one call picks the best item for each key kind
   (period, region, sample, source, method); a second, narrower call picks
   method facts that apply to all results. These reach every statistic the
   study reports itself, but not figures cited from other studies. A vague
   period, sample, collection method or region gives way to the one stated
   on the method's page, and a key kind the scope step left out is taken from
   that page.
3. **Statistics** per page, then a **linking** call per page for the
   specific items each number needs (its survey question, scenario, source).
4. **Plain-code rules**: the reading the wording states ("just over" is a
   lower bound), values as printed ("80%" is 80, "half" is 50%), the period
   nearest the value in its sentence at printed precision, survey answers
   dated to the fieldwork year, citations make a figure "cited". Table rows,
   chart and infographic labels, reference lists, date-only values and
   numbers that only set up the report are dropped: subgroup definitions and
   bounds ("6-10 people", "(5+ days)"), scale ends, counts of the report's own
   parts ("three main themes", "three production seasons"), thresholds in a
   rule ("if case numbers are below five", "the 1% cut-off") and period
   lengths ("the five-year period"); a bracketed unit conversion and the two ends of a printed range
   are kept as one statistic.
5. **Context items that are not conditions** are dropped before scoping and
   linking: running headers and footers, thanks, disclaimers and copyright
   lines, and values that are only a generic word ("Survey", "Australians").
   Study items are then fitted to each statistic: a model input keeps only the
   study's period, region and data source, and a rating scale reaches only
   survey answers (a 1 to 5 scale without agreement wording only the ratings).
6. **Duplicates**: numbers with the same value and unit are paired (also a
   value restated as a bound, "approaching" and "approximately $3.5 million",
   or rounded, with units in any word order), and one
   model call per document judges which pairs are the same measurement (the
   abstract repeating a result). The later one is dropped and the earlier
   keeps both sets of links; different survey questions or groups are kept
   apart. `--no-dedupe` turns this off.

```bash
.venv/Scripts/python.exe scripts/statistics/run_statistics_extraction.py             # reference documents, local E4B
.venv/Scripts/python.exe scripts/statistics/run_statistics_extraction.py --score-only output/statistics/<run>
```

Scored against `data/reference/statistics.json` and `contexts.json` (four
documents, 239 statistics), the latest E4B run finds 99% of the reference
statistics (237) at 96% precision. Field agreement on matched statistics:
value reading 99%, type 92%, period 90%, record type 87% (repeat runs vary by
a few points). Context links reach 85% of the facts the reference links, and
57% of the links given are right. Key conditions are right for nearly all
statistics: region 204 of 207, time period 156 of 157, sample 102 of 109, data
source 100 of 118. A
run takes about 2 to 7 minutes per document.

Context links are scored two ways: recall by needed fact (each fact the
reference links to a statistic), precision by link (a link is right when its
fact is needed or acceptable for the statistic; a second link to the same right
fact is redundant, not wrong). A reference statistic's `acceptable_item_ids`
are facts that are fair to link but not needed, such as a model assumption
behind an LCA result. The reference was checked link by link against a reading
of the PDFs (`output/verification/context_audit.md`); the two now agree on all
but a few dozen links where quotes happen to overlap.

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

Both extraction scripts read such pages from the page image by default
(`--no-vision` turns this off): for statistics, any page with two or more
unreadable characters, garbled text or no text; for metadata, only garbled or
empty pages. Whole pages are read rather than crops of the unreadable words:
on Voconiq a word crop took about 1.3 seconds against about 2.8 for a page,
flagged pages hold around ten such words, and small crops were read less
reliably.

Flagged or empty pages can also be transcribed on their own:

```bash
.venv/Scripts/python.exe scripts/ingestion/transcribe_unreadable_pages.py
```

Only running text is transcribed; charts and tables become `[figure]`
because they belong to the table/image workstream. Gemma 4 E4B reads a page in
about 3 seconds (12B takes 7) with the same accuracy; output goes to `output/vision/`.

## Checking the reference

```bash
.venv/Scripts/python.exe scripts/statistics/verify_ground_truth.py   # reference statistics against the PDFs
```

## Layout

```
src/crdc_pipeline/
  ingestion/
    layout.py        column detection (pure function, no PDF library dependency)
    loader.py        PDF -> Document/Page; text, symbol repair and image fallback
    symbols.py       finds wrongly labelled and hidden characters, learns font rules
  extraction/
    backends.py      Gemini and Ollama backends, one interface (shared)
    vision.py        page and word reading from images (shared)
    metadata/
      extract.py     metadata schema, prompt, extraction and tidy rules
      printed.py     fixed-layout details (article history, DOI, journal header, citation, emails)
      names.py       standard organisation names
      scoring.py     field matching against the reference
    contexts/
      schema.py      context items and contexts, and grouping statistics into contexts
      prompts.py     prompts for context items, the study's scope and linking
      scope.py       the study's own conditions: labels dropped, vague items swapped for the method's
      links.py       trimming repeated links, and the whole-document linking pass
      scoring.py     context items and links against the reference
    statistics/
      extract.py     the pipeline: context items, scope, statistics, links, then the rules
      schema.py      statistics
      prompts.py     prompts for statistics and duplicates
      filters.py     what to leave out: reference lists, table rows, set-up numbers
      values.py      each value's reading, bounds, type and labels
      periods.py     periods at the precision printed
      duplicates.py  restatements, conversions in brackets and ranges
      records.py     record types decided across the document
      scoring.py     statistics against the reference
  reference.py       loads the references in data/reference/
scripts/
  ingestion/transcribe_unreadable_pages.py   re-reads flagged pages with a vision model
  metadata/run_metadata_extraction.py        runs metadata extraction and scores it
  statistics/run_statistics_extraction.py    runs statistics and context extraction and scores both
  statistics/verify_ground_truth.py          checks the older ground-truth statistics against the PDFs
tests/
  conftest.py                          reads the sample PDFs once, to check reference quotes
  test_backends.py                     backends with mocked HTTP (no server needed)
  ingestion/                           column detection, loading (synthetic PDFs), symbols, the sample PDFs
  metadata/
    test_metadata_extraction.py        schema, prompt and tidy rules
    test_printed.py                    article history, DOI, journal header, ISSN
    test_metadata_scoring.py           field matching
  contexts/
    test_context_schema.py             grouping statistics into contexts
    test_context_rules.py              the study's scope, labels, item periods and links
    test_context_scoring.py            context item and link scoring
    test_contexts_reference.py         reference items fit the schema and are quoted from their page
  statistics/
    test_statistics_pipeline.py        the whole pipeline with a fake model
    test_statistics_rules.py           readings, types, periods, filters and duplicates
    test_statistics_schema.py          statistic schema
    test_statistics_scoring.py         statistic scoring
    test_statistics_reference.py       reference statistics fit the schema and are quoted from their page
data/
  raw_pdfs/                        the four source PDFs (gitignored)
  document_manifest.json           document_id, filename and download link
  reference/
    metadata.json                  reference metadata, 19 fields per document
    statistics.json                reference statistics, with the context items each one needs
    contexts.json                  reference context items and contexts
  schema_worked_examples.json      worked examples of the documents, contexts and statistics tables
  ground_truth_all_documents.json  older 48-statistic reference set
  organisation_names.json          standard name and known variants per organisation
  symbol_rules.json                learned corrections for wrongly labelled symbols
docs/
  weaknesses_and_fixes.md          known extraction problems and planned fixes
  GitHelper.md                     git workflow for the team
  architecture.md                  design notes for the ingestion module
```
