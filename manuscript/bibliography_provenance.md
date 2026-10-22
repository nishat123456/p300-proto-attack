# Bibliography provenance and remaining citation work

22 October2026. references.bib contains ten DOI-registered journal/conference
records fetched from Crossref and four separately identified arXiv preprints.
Compact registration fields are retained in bibliography_metadata.json; records
are not evidence that the paper's methods were read. Source-specific reading
scope is documented in results/literature_protocol_matrix_20261022.md.

Crossref requests used https://api.crossref.org/works/DOI and its
/transform/application/x-bibtex route. Preprint titles/authors/identifiers were
checked against their arXiv abstract records. Journal and preprint Chen entries
are distinct keys, not interchangeable publication versions. Method-source
preprints for BU/KDE, LID and Mahalanobis are included without inventing venue
metadata. EEGNet DOI10.1088/1741-2552/aace8c is distinct from the unrelated
EEGNET connectome tool.

Publication-date conventions require care. TinyNoise was published online in
September2020 but appears in NSR8(4), April2021. Its BibTeX uses issue year2021,
with the online date retained in a note and article identifier nwaa233. This is
verified against the publisher issue page https://academic.oup.com/nsr/issue/8/4
and article https://academic.oup.com/nsr/article/8/4/nwaa233/5903729.
Other month fields were omitted because DOI registration month can differ from
online/publication dates; no exact publication-day claim is based on those fields.

AUAP DOI contains2024 while its ESWA268 issue is2025; use the publisher's2025
issue year. GPN author spelling follows Crossref and PubMed37556336, not an
inconsistent secondary index. Expanded JNE authors follow its publisher title
page and registration; the author website's extra Bian is not copied.

These are working citation records, not a journal-formatted final manuscript.
Next add verified dataset, calibration/statistical and adaptive-evaluation
method sources where used, integrate citation keys into the final authoring
format, check publisher capitalization/diacritics and compile the bibliography.
Do not infer exact reproduction, source-code license or novelty from a citation.
No experimental code, protocol, threshold or result changed during this work.
