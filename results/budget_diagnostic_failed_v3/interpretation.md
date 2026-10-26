# Diagnostic packaging correction, 26 October 2026

Version3 again failed during cloud transfer of task00 calibration JSON before strict numerical audit. Static metadata inspection found enable_internet=false, retained from the original failed-kernel-mount-only diagnostic. This is a confirmed packaging defect for a direct-download runner. The previous expiry hypothesis was not established; refreshed links alone did not resolve the failure. No numerical or performance findings were produced.

Version4 enables internet, retains the same strict budget audit, refreshes authenticated cloud output links and adds bounded failure metadata (exception class and HTTP status without URLs). No scientific source, threshold, raw epsilon or assertion changes. Original output artifacts and all failed runs remain preserved. Scientific validity and original strict budget violation remain unresolved until numerical evidence is available.
