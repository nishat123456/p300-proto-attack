# Fixed clean validation, October15,2026

Private Kaggle p3a-openbmi-valclean v1 completed all7methods/3seeds at fixed
subject19,36 official test decisions per seed. Twenty separate official-train
calibration decisions set thresholds;13 other train decisions set normalization.
No thresholds, model seeds or methods were chosen from these outcomes. Job's29
integration tests passed13.70s. Compact JSON/log only retrieved; per-character
CSV/model/reference/raw artifacts remain on Kaggle.

All3seeds reported clean accuracy36/36. This is one development subject, not108
independent characters or3 independent people. At nominal5%, rejected counts:

|Method|20261013|20261014|20261015|
|---|---:|---:|---:|
|Confidence|0/36|0/36|0/36|
|Entropy|0/36|0/36|0/36|
|BU|0/36|2/36|1/36|
|LID|1/36|0/36|2/36|
|MD|0/36|2/36|3/36|
|Adjusted MD|0/36|2/36|3/36|
|KDE|1/36|1/36|0/36|

Observed0/36,1/36,2/36,3/36 correspond to0%,2.78%,5.56%,8.33% respectively.
Keep all nominal1/5/10% results and accepted-wrong/accepted-correct/IID intervals
in the original JSON, not just this primary nominal5% view. No clean errors
were observed here, so this subject cannot demonstrate behavior on genuine
mistakes. Zero false alarms still has a two-sided95% IID upper bound9.74%, and
serial EEG decisions need not be IID. No precise5% assurance follows.

These outcomes show fixed calibration can be executed and has seed/method
variation. They do not measure attack TPR, defended attack success, unseen-user
reliability, subject-population uncertainty or Q1 readiness. Do not select
confidence/entropy as confirmed winners or retune MD using these test results.
Next verify real attack paths and fixed optimization behavior on development
inputs, then complete/freeze protocol/manifests before sealed outcomes. Historical
gates/splits remain unchanged; BNCI2015-003 locked.
