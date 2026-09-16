# Metadata extraction: design notes

Covers `src/crdc_pipeline/extraction/metadata.py`. Companion to
`docs/architecture.md` (which covers ingestion) and
`docs/weaknesses_and_fixes.md` (the statistics-extraction problems this
module doesn't attempt to solve).

## Schema decisions from the 9 Sept 2026 project meeting

Running metadata extraction against the four sample documents surfaced
three fields where the right schema wasn't obvious from the brief
alone: organisation, publication date, and source URL. Raised with
Burton in the 9 Sept meeting; his guidance below is what the current
schema implements.

### Organisation, publisher, and funder are three different fields

A single `organisation` field was ambiguous: Gemini's output assigned
different organisations inconsistently across documents (AgHealth
Australia for aghealth2019, CRDC for aghealth2024) because a report
commonly involves an author's institution, a publisher, and a funder,
and nothing told the model to tell them apart or which one to prefer.

Burton's guidance: `organisation` should specifically mean **author
affiliation**. Publisher and funder are separate categories entirely,
not alternatives to fall back on when the "main" organisation is
unclear. The schema now has three fields (`organisation`, `publisher`,
`funder`), and the extraction prompt explicitly names all three and
instructs the model to fill in more than one when the text supports
it, rather than forcing a single choice.

(Separately: AgHealth Australia describing itself as part of The
University of Sydney is a real division-of-a-university relationship,
not a misattribution -- confirmed with Burton, no schema change needed
for that case specifically.)

### Multiple dates need their own fields

Some documents state more than one date (Hedayati's journal issue year
is 2019, but its actual submission date is 2018-11-22). Burton offered
two options: a nested JSON structure holding every date version, or a
flat structure with separate named fields (issue date, submission
date, etc.), noting that the main use of the date downstream is simply
judging how current a document is.

Went with the flat option, consistent with the rest of this schema:
`publication_date` stays the primary field (the one that answers "how
new is this"), with optional `submission_date` and `acceptance_date`
alongside it for documents that state them separately. A document
that only gives one date just fills `publication_date` and leaves the
other two null.

### source_url stays in the schema even when it's always null here

None of the four sample documents have an in-text URL, which raised
the question of whether the field was worth keeping. Burton's answer:
keep it regardless -- future documents or ingestion sources may
provide one, and the schema shouldn't have to change shape just
because this particular corpus happens not to need it.

## Finding: the Gemini web interface is not reliable for this task

Documented per Burton's explicit request, since he called this "a good
finding" worth recording rather than something to just work around
quietly.

Before writing `metadata.py`, the same extraction schema was tried
through Gemini's web chat interface directly (with the desired JSON
output structure described in the prompt). It did not reliably return
the fields asked for, and appeared to apply its own assumptions about
what to include even when the target structure was given explicitly.
The API-based approach in this module (structured output via
`response_schema`, `temperature=0`) does not have this problem --
every one of the six fields verified against the reference metadata in
`schema_worked_examples.json` for aghealth2019 matched exactly (see the
git history for `scripts/run_metadata_extraction.py`'s comparison
output).

The likely reason (per Burton): chat interfaces are tuned for
conversational use and can apply their own additional instructions or
guardrails on top of a user's prompt, which is exactly the kind of
interference that breaks an exact-format structured-extraction task.
**Conclusion for this project: the web interface should not be used to
test or validate structured extraction quality. Testing needs to go
through the API**, as this module already does.

## Verification: the organisation/publisher confusion is fixed

Re-running extraction against all four documents after the schema and
prompt changes above:

- **Hedayati**: `organisation` now correctly returns "NSW Department of
  Primary Industries" (matching the reference exactly) instead of
  "Elsevier Ltd.", with `publisher` separately holding "Elsevier".
  `submission_date` (2018-05-19) and `acceptance_date` (2018-11-21) are
  now captured distinctly from `publication_date` (2019, the journal
  issue year) -- the acceptance date lands within a day of the
  reference's single `publication_date` value (2018-11-22), which is
  itself evidence the reference was using the acceptance date where
  this schema now uses the issue year for `publication_date` by
  design (see "multiple dates" above).
- **AgHealth 2019**: `funder` now correctly holds "Cotton Research and
  Development Corporation" as its own field, no longer at risk of
  overwriting `organisation` (which stays "AgHealth Australia (The
  University of Sydney)", matching the reference).
- **VoconiQ**: `publisher` and `organisation` both correctly resolve to
  "Voconiq"; the reference's single combined string ("Voconiq,
  commissioned by Cotton Australia and Cotton Research and Development
  Corporation") is exactly the kind of multi-role case this schema
  split is meant to disambiguate, not something to reproduce verbatim.

## Known gap: schema_worked_examples.json predates this decision

`data/schema_worked_examples.json` (the team's shared reference schema,
provided at project start) still represents each document's
organisation as a single string -- e.g. VoconiQ's entry is `"Voconiq,
commissioned by Cotton Australia and Cotton Research and Development
Corporation"`, one field doing the work that `organisation` +
`funder` now do separately in this module's schema. This module's
richer schema is a superset (it can represent everything the reference
file's flatter one does, just split up), but the two aren't the same
shape. Whether the shared reference schema itself should be updated to
match is a team decision, not something this module changes
unilaterally -- flagging it here rather than editing that file.
