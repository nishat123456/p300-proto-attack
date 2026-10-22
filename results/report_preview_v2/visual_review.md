# Development figure visual review, 22 October 2026

Private p3a-report-preview v2 completed22.2s with actual notebook image output.
Browser inspected Notebook version2 on Kaggle, using two scroll views covering
both existing PNGs. No PNG was downloaded or recomputed locally.

`clean_rejection.png`: one participant19 row, seven readable detector labels,
participant axis, development-only title, nominal5% label and fixed0-100%
colorbar are visible. Labels do not overlap the plotted data or colorbar.

`defended_manipulation.png`: all seven detector rows and five attack columns,
including detector-aware, are visible across the inspected views; budget0.20,
all-attempt denominator, development-only title and fixed0-100% colorbar are
readable. Axis labels fit the image. Separate detector-aware objectives must
remain explicit in the accompanying caption; this is not simultaneous evasion.

This verifies the displayed development figure layout only. It does not verify
future17-participant figure layout or add population evidence. Input PNG hashes
are bound in out/preview_provenance.json; original generated report provenance
retains its historical pending-review statement, superseded for these two exact
images by this separate review record. No threshold or experimental code changed.
