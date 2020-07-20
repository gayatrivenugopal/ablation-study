# Ablation Study

This code will help performa a basic ablation test.
Constraints:
1. One feature will be removed at a time.
2. We are assuming a classification task.
3. The metrics returned are accuracy, macro-f1 and AUC scores

Inputs:
training records, training target, test records, test target

ablation_analysis.py eliminates one feature at a time and assesses the performance.
metrics.py contains the code to calculate Accuracy, Macro-F1 and AUC score.
models.py contains the code to instantiate the classifiers. Use this file to add more models.

