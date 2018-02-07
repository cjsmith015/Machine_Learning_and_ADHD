# Machine Learning & ADHD

_Christiana Smith - Galvanize, Inc., Data Science, February 2018_

# Project Background
Supervised and unsupervised learning are powerful statistical tools that are surprisingly absent in clinical and psychological research. With this in mind, I collaborated with the **ADHD Research Lab** at the **Oregon Health and Science University** to show what data science can do for their lab and research.

# The Data

# Questions

# Technical Process
I have documented my entire process in the Jupyter Notebooks, 'Supervised Learning' & 'Unsupervised Learning'. If you wish to run this project on your local machine, read the instructions below.

## Prerequisites
The following Python libraries are used:
 * Pandas
 * NumPy
 * scikit-learn
 * SciPy
 * fancyimpute [link](link)
  * Utilizes keras, Tensorflow backend
 * xgboost [link](link)

## Dataset Preparation
The original csv `Christie_diagnosis_20180118.csv` is housed in a directory `data`, not included on the github repo. It must be split into a training and holdout dataset. This is done by running `holdout_set_prep.py` which splits the original csv into `holdout_data.csv` and `train_data.csv` at a test size of 33%.

# Supervised Learning

To answer the question of which machine learning models best predict ADHD diagnosis, I obtained model metrics on four models (logistic regression, random forest classifier, gradient boosting classifier, and xgboost classifier), on four datasets (DX~All, DXSUB~All, DX~TMCQ, DX~Neuro).

## Data Prep
### Missing Value Imputation
Through some analysis, I determined that **Matrix Factorization** was the best strategy for missing value imputation.

### Data Leakage Check
I examined each variable and concluded none were at risk for data leakage.

## Model Metrics

## Tuning The Models

## Conclusions

# Unsupervised Learning

## Conclusions

# Built With

# Acknowledgments
