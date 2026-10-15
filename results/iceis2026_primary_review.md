# ICEIS2026 primary protocol review

Verified October15,2026. DOI10.5220/0014819600004018. Publisher PDF retrieved:
https://www.scitepress.org/Papers/2026/148196/148196.pdf . Sections3.2-4 and
Table2 read; pages884/885/887 visually checked. Full PDF is not copied here.

The study uses GrazA motor-imagery EEG, nine participants and EEGNet, with
7-35Hz preprocessing and200Hz sampling. It evaluates FGSM, PGD, DeepFool and
CW; the detection comparison includes RF/SVM/KNN and ANFIS. Detection uses a
stratified80/20 train/test split. ANFIS inputs include embedding magnitude,
output entropy, confidence and logit margin. Table2 reports FGSM detection;
the text also discusses more difficult DeepFool detection.

The reviewed protocol does not specify an independently calibrated P300
character-decision FAR/rejection-cost experiment. Do not infer unreported
implementation details are absent, or compare its accuracy numbers directly
to our character-level metrics. This is an additional conventional/uncertainty-
aware EEG detection precedent, not support for a first-detector claim.

Our candidate gap remains independently calibrated character reliability,
user shift and cost, conditional on substantive independent evidence. This
review closes the earlier full-protocol-access uncertainty for this paper;
it does not establish exhaustive novelty or justify adopting ANFIS from
outcome comparisons. No new method/question or performance setting is selected.
