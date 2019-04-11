# Procssesing code and results for our PE paper
# main.m       
The whole processing code, including data reading, 100 times 10-folds CV (CV-LASSO + LSVM), results recording and ROC curve plot. By running this file, some new files, including IDX_FS.mat (The selected features indice by CV-LASSO under different parameters), group_test.mat (The subject's index in testing group in 100 times 10-folds CV), index.mat (model performance indice under different parameters, including accuracy, sensitivity, specificity and recall), weights.mat (features weight in classification) and edge*.txt (ROI1-ROI2), will be created.
# statistic.m  
The results recording and ROC curve plot part of main.m. By running this file, some new files, weights.mat (features weight in classification) and edge*.txt (ROI1-ROI2), will be created. 
# Features_all or Feature_BN246.mat
Including correlation cofficients between 90 ROIs in aal atlas or 246 ROIs in BN 246 atlas
# names.mat
Including the name of aal 90 ROIs and BN246 246 ROIS, extracting from CONN software
# Update Apr 11, 2019
update the permutation test code and fix errors of ROC relevant in main.m and statistic.m
# permutation.m
Permutation test of accuracy and AUC
