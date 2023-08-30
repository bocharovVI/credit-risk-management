# Clients binarized data

This is a classification problem for bank's clients. There're twelve parquet files which contains inforamation about customer behavior and their credit products.


## Files structure


* Preprocessing.ipynb - data preparation
* *_model.ipynb - test models
* Results.ipynb - compare different models


data/

* train_data/ - parquet files
* train_target/ - csv targets
* processed_data/ - processed parquets
* processed_data_small/ - small part of processed parquets
* ensemble_models/ - saved models of ensemble model class 
* single_models/ - saved single type models
* scores.csv - dataframe with results


scripts/

* preprocess.py - data preparation function
* Ensemble.py - ensemble model class
* pipeline.py - pipeline script


pictures/ directory contains graphics and other visualizations.


## Encoding data


The dataset was encoded part by part by OneHotEncoder from sklearn libraly and aggregated by id for sum. For Logistic regression, Random forest, Single MLP pytorch, Catboost and LightGBM all processed parquets loaded and stacked at single dataframe with all possible features. 


## Results


In this project 6 machine learning models were tested: Logistic regression, Random forest, Single MLP pytorch, Ensemble MLP pytorch, Catboost, LightGBM. Tests have shown that the best result is shown by the LightGBM model. It shows a good result of the AUC metric on the test set and takes minimal time.
