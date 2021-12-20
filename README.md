# Predicting symptoms of downy mildew, powdery mildew, and gray mold diseases of grapevine through machine learning
This repository is related to the the paper “Predicting symptoms of downy mildew, powdery mildew, and gray mold diseases of grapevine through machine learning” authored by Iride Volpi, Diego Guidotti, Michele Mammini, and Susanna Marchi, submitted to the Italian Journal of Agrometeorology on 10 November 2020.
The repository contains, as an example of the method applied in the paper, a script and an extract of the dataset used for training, tuning and testing the model for the prediction of the presence of symptoms of powdery mildew on grapevine in Tuscany region (Italy). 

## Description of the files
# R script
The R script contains all the steps of the methodology adopted in the paper to predict the presence of symptoms of downy mildew, powdery mildew, and gray mold on grapevine in Tuscany Region (Italy). For each disease, the presence of symptoms was predicted with machine learning using a set of variables associated with the infestation (i.e., bioclimatic indices, geographical indices, the number of crop protection treatments during the growing season and the frequency of monitoring days in which symptoms were recorded in the previous year).
The R script includes the following steps of data analysis:
* Dataset partition
* Selection of variables
* Training and selection of the classifiers
* Predictions on the test sets
* Variable importance and partial dependence plots

# Example dataset
A subset of the entire dataset used in the paper for the model on powdery mildew was provided. The dataset contains observation of the symptoms of powdery mildew for a set of vineyards (farm_ID) and the associated variables.
The variables were calculated according to what reported in the section 2.2 of the paper.

