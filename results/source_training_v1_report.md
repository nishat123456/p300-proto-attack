# Source-only decoder training completion, October15,2026

Private Kaggle p3a-openbmi-source v1 completed all three fixed seeds20261013,
20261014,20261015. Each source record reports30 epochs,35640 flash epochs,
positive class weight5.0, final-epoch checkpoint policy and source subjects1-18
only. Config hash0cae4a658b58877541b7289905ea6d81c710b1de9738fb97903e87ebd9d7fb24
and schedule hash9b7de951b9867e546577cae72f596004ff7dca4a97f941e9e9a3578767149230
match across seeds. Source payload381ac26, exact launcher543cebf. Job's initial
19 portable tests passed17.88s; subsequent broad suite351 passed is preserved
separately in results/portable_suite_v14, with explicit exclusions.

Only compact JSON/log/test output was retrieved. Raw EEG and checkpoints remain
on Kaggle. No source-model accuracy, clean FAR, attack success, TPR or independent
outcome is inferred from training-loss completion. No best seed was selected.

Next staged p3a-openbmi-refcal consumes the completed own-source kernel in cloud,
fits source references and the template adaptation, then calibrates fixed
validation subject19 across all3seeds. That end-to-end job is not yet verified.
Independent protocol freeze/manifests and sealed evaluation remain pending.
