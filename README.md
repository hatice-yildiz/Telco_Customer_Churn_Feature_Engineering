# Telco_Customer_Churn_Feature_Engineering

Data analysis and feature engineering steps 
were performed to develop a machine learning model 
that can predict customers who will churn the company.


TASK 1: EXPLORATORY DATA ANALYSIS
           # Step 1: Examine the overall picture.
           # Step 2: Catch numeric and categorical variables.
           # Step 3: Analyze the numeric and categorical variables.
           # Step 4: Analyze the target variable (average of the target variable with respect to categorical variables, average of numeric variables with respect to the target variable)
           # Step 5: Analyze outlier analysis.
           # Step 6: Perform missing observation analysis.
           # Step 7: Perform correlation analysis.

 TASK 2: FEATURE ENGINEERING
           # Step 1: Perform the necessary operations for missing and outliers.
           # Step 2: Create new variables.
           # Step 3: Perform encoding operations.
           # Step 4: Perform standardization for numeric variables.
           # Step 5: Create a model.

Required Library and Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")
