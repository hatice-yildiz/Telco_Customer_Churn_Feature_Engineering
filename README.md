<p>Data analysis and feature engineering steps were performed to develop a machine learning model that can predict customers who will churn the company.</p>

<h3>TASK 1: EXPLORATORY DATA ANALYSIS</h3>
<ol>
  <li>Step 1: Examine the overall picture.</li>
  <li>Step 2: Catch numeric and categorical variables.</li>
  <li>Step 3: Analyze the numeric and categorical variables.</li>
  <li>Step 4: Analyze the target variable (average of the target variable with respect to categorical variables, average of numeric variables with respect to the target variable)</li>
  <li>Step 5: Analyze outlier analysis.</li>
  <li>Step 6: Perform missing observation analysis.</li>
  <li>Step 7: Perform correlation analysis.</li>
</ol>

<h3>TASK 2: FEATURE ENGINEERING</h3>
<ol>
  <li>Step 1: Perform the necessary operations for missing and outliers.</li>
  <li>Step 2: Create new variables.</li>
  <li>Step 3: Perform encoding operations.</li>
  <li>Step 4: Perform standardization for numeric variables.</li>
  <li>Step 5: Create a model.</li>
</ol>

<h3>Required Library and Functions</h3>

<pre><code>import numpy as np
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
</code></pre>
