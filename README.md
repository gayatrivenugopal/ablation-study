# Ablation Study

This code will help performa a basic ablation test.<br>
Constraints:<br>
1. One feature will be removed at a time.<br>
2. We are assuming a classification task.<br>
3. The metrics returned are accuracy, macro-f1 and AUC scores<br>
<br>
Inputs:<br>
training records, training target, test records, test target<br>
<br>
ablation_analysis.py eliminates one feature at a time and assesses the performance.<br>
metrics.py contains the code to calculate Accuracy, Macro-F1 and AUC score.<br>
models.py contains the code to instantiate the classifiers. Use this file to add more models.

