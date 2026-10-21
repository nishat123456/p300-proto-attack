Source-only timing v1 passed363 portable tests then failed LID zero-distance guard
on source1 first train query, which can occur in its source reference bank.
Timing-only fix excludes that exact query's flash identities explicitly, not by
observed distance. No epsilon floor, detector assertion change, query selection
or independent reference/statistical outcome changes. Rerun pending.
