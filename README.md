# Diabetes-prediction
Diabetes prediction based on plasma biomarkers

The aim of the project - predict diabetes incidence based on 36 plasma biomarkers (predictors, G2-G37). Age, smoling status and ethnicity were treated as confounders.


1) Predictors data preprocessing steps can be found in html format - 1_Predictors_data_processing.html

I removed outliers with a thershold of 5%. Since, outliers can be of biological significance I saved both datasets with outliers and without them, and analysed them separately. Outliers removal decreased accuracy of prediction (see item 4)


2) Analyses of the data with removed outliers in pdf format - 2_Model-comparison-and-analyses-of-effects_outliers-removed.pdf
The best achieved accuracy for test - 0.731


3) Analyses for the data where outliers were present - 3_Model-comparison-and-analyses-of-effects_all_samples.pdf
The best achieved accuracy for test - 0.751 (GLM model with all predictors used)

4) Script for a deep neural network trainings - 4_DNN_training.py

5) Example of a bash script to run over various parameters of a DNN - 5_dnn_layers_var.sh 
The best achieved accuracy for test for DNN model - 0.735. The reason for that is likely to be the limited sample of patients. 
