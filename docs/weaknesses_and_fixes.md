# Frontier-Model Benchmark, Weaknesses and Planned Fixes

Ground truth extraction across the four CRDC sample documents identified 48 prose-based statistics (13 AgHealth 2019, 10 AgHealth 2024, 12 Hedayati 2019, 13 VoconiQ 2025). This document records the eight weaknesses that extraction surfaced, ranked roughly by how much they'd affect a local model, and the fix planned for each.

## 1. Two-column layout breaks position-preserving text extraction (Hedayati 2019)

`pdftotext -layout`, which tries to preserve on-page position, merges the two columns of the Hedayati journal article line by line, splicing unrelated sentences from each column together on the same output line. The same tool without `-layout` reads the columns correctly in order.

**Fix:** detect column count before extraction (single vs two column), and route two-column documents through plain `pdftotext` or a layout-aware library rather than position-preserving mode. This needs to happen before any LLM sees the text, prompting cannot recover a scrambled reading order.

## 2. Certain numeric glyphs are missing from the text layer entirely (VoconiQ 2025)

Four values that are clearly visible in the rendered page (41%, 28%, and a decimal mean score) are completely absent from the extracted text, confirmed independently in both `pdftotext` and `pdfplumber`. Since two different extraction libraries fail identically, the fault is in the source PDF's font encoding, not the tool. No amount of prompting can retrieve text that was never extracted.

**Fix:** this needs a fallback path, either OCR on the rendered page image, or a multimodal model reading the page directly as an image rather than as extracted text. Worth scoping how common this is across the wider CRDC corpus before deciding whether to build it now or flag it as a known limitation.

## 3. Statistics needing a cross-reference elsewhere in the document to interpret correctly

AgHealth 2019's "two additional cases were identified... during this period" only resolves to a specific time period by reading a separate paragraph several sentences later. A model extracting sentence by sentence will get the number right and the period wrong.

**Fix:** two-pass extraction, first pass surfaces candidate statistics, second pass re-checks each one against the full document for period or subject disambiguation, rather than one-shot per-sentence extraction. Schema needs a way to flag which statistics required this.

## 4. Hedge and approximation language

Both AgHealth reports use "surpasses," "approaching," "just over," and tilde-prefixed percentages routinely, for money figures, claim counts, and percentages alike. Treating these as exact numbers overstates precision that the source itself doesn't claim.

**Fix:** `value_status` field distinguishing exact, approximate, rounded, and censored, with the exact wording preserved in a `reported_text` field regardless of status.

## 5. Suppressed and rounded values under CRDC's own data agreements

AgHealth 2024 explicitly withholds exact figures below five ("fewer than five cases") and separately rounds recent workers compensation figures to the nearest five, both by design, not by omission.

**Fix:** `censored_below` as its own `value_status`, with `value` set to null and `reported_text` holding the source's actual wording, so this is recorded as an intentional data governance decision rather than a failed extraction.

## 6. Report revisions don't align cleanly across document versions

AgHealth 2024 updates 2019's figures, but not on a clean like-for-like basis, the 2001-19 total of 17 fatalities at $48 million (2018 dollars) becomes 21 fatalities at $73 million (2023 dollars) in the 2024 report, covering 2001-23. The case count, the period, and the currency year all changed at once, so a simple "this revises that" link risks implying more precision than actually exists.

**Fix:** `supersedes_statistic_id` should only be used when the period and units genuinely match, everything else gets a plain note explaining what changed instead of a false direct link.

## 7. Model assumptions look identical to measured findings

Hedayati's "96% of production was from irrigated systems" is an input the researchers assumed for their model, not something they measured. Extracted without distinction, it reads exactly like the paper's actual results.

**Fix:** `record_type` field, `measurement`, `model_assumption`, `scenario_result`, or `survey_response`, so a `scenario_result` like "2.6% emissions reduction" is never read as a real-world outcome rather than a modelled one.

## 8. Repeated context duplicated across related statistics

VoconiQ's water responsibility question, its exact wording, methodology, and sample, is the same for both the 2025 and 2020 figures tied to it. Duplicating that context per statistic gets unwieldy fast across a document with dozens of survey items.

**Fix:** shared `context` object referenced by multiple statistics, per the schema already discussed with Dr Wu.

---

## What this means for the completeness estimate

The 48 statistics above are the reference set for measuring recall. To get the completeness figure Dr Wu asked for, run the local model over these same four documents and calculate:

**(statistics the local model correctly extracts, matching value and time period) / 48**

We can't produce that percentage here, it requires running an actual local model, which needs to happen on the team's own setup. This document and the accompanying ground truth file are the fixed reference point to measure that run against.

## What still needs local model testing specifically

- Speed and token cost per document, compared to frontier model calls
- Whether prompting page by page versus whole document changes recall, particularly for the two-column Hedayati paper once the layout fix is applied
- Whether a smaller model reproduces the same failure modes identified here, or introduces new ones
