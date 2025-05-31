 # decision_tree

# Decision Tree Assignment

## About the Data
The dataset provided for this assignment pertains to a company's sales records, encompassing approximately 400 records and 10 variables. Here are the attributes included:

- **Sales:** Unit sales (in thousands) at each location.
- **Competitor Price:** Price charged by competitor at each location.
- **Income:** Community income level (in thousands of dollars).
- **Advertising:** Local advertising budget for the company at each location (in thousands of dollars).
- **Population:** Population size in the region (in thousands).
- **Price:** Price company charges for car seats at each site.
- **Shelf Location at Stores:** A factor with levels Bad, Good, and Medium indicating the quality of the shelving location for the car seats at each site.
- **Age:** Average age of the local population.
- **Education:** Education level at each location.
- **Urban:** A factor with levels No and Yes to indicate whether the store is in an urban or rural location.
- **US:** A factor with levels No and Yes to indicate whether the store is in the US or not.

## Problem Statement
A cloth manufacturing company aims to identify the segment or attributes that cause high sales.

## Approach
A decision tree will be constructed with the target variable "Sales" (converted into categorical variable) and all other variables being independent in the analysis.

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import ppscore
Dataset Understanding

# Importing dataset
com = pd.read_csv('Company_Data.csv')

# Checking for information of the data
com.info()

# Checking for null values
com.isna().sum()

# Checking the correlation
com.corr()

# Unique values in 'Sales'
com.Sales.unique()

# Columns in the dataset
com.columns

# Grouping by different attributes
com.groupby(['ShelveLoc','Urban','US']).count()
Data Preprocessing
# Label Encoding
label_encoder = preprocessing.LabelEncoder()
com['ShelveLoc']= label_encoder.fit_transform(com['ShelveLoc'])
com['Urban']= label_encoder.fit_transform(com['Urban'])
com['US']= label_encoder.fit_transform(com['US'])

# Converting categorical variables to category type
com['ShelveLoc']=com['ShelveLoc'].astype('category')
com['Urban']=com['Urban'].astype('category')
com['US']=com['US'].astype('category')

# Checking data types and info
com.info()
Data Visualization

# Heatmap for correlation
plt.figure(figsize=(20, 8))
sns.heatmap(com.corr(), cmap='magma', annot=True, fmt='.3f')
plt.show()

# Pair plot to visualize the data
sns.pairplot(data=com)

# Regression plot
sns.regplot(x='Sales', y='Income', data=com, color='black')

# Bar plot for 'ShelveLoc'
com.ShelveLoc.value_counts(ascending=True).plot(kind='bar')
Decision Tree Modeling

# Splitting data into features and target variable
x = com.drop(['Sales'], axis=1)
y = com['Sales']

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
Conclusion
This README provides an overview of the dataset, problem statement, approach, data preprocessing steps, visualization, and decision tree modeling for the given assignment.

## Code README

This README provides an overview of the code for predicting on test data and building decision tree classifiers and regressors using different criteria.

### Predicting on test data

- **Predictions:** The code predicts the target variable on the test dataset.
- **Value Counts:** Counts the occurrences of each category in the predictions.
- **Two-way Table:** Provides a two-way table to understand correct and wrong predictions.
- **Accuracy:** Calculates the accuracy of the predictions.

### Building Decision Tree Classifier (CART) using Gini Criteria

- **Model Creation:** Builds a Decision Tree Classifier using the Gini impurity criteria.
- **Prediction and Accuracy:** Predicts on the test data and computes the accuracy of the model.

### Building Decision Tree Classifier using Entropy Criteria

- **Model Creation:** Constructs a Decision Tree Classifier using the Entropy criteria.
- **Prediction and Accuracy:** Predicts on the test data and computes the accuracy of the model.

### Problem 2 (Fraud_Check)

- **Problem Description:** Uses decision trees to prepare a model on fraud data, treating individuals with taxable_income <= 30000 as "Risky" and others as "Good".
- **Data Understanding:** Describes the dataset and checks for null values.
- **Data Preprocessing:** Handles categorical variables, creates dummy variables, and bins the taxable income column.
- **Normalization:** Normalizes the data for modeling.
- **Model Building:** Builds a Random Forest Classifier and a Decision Tree Classifier on the fraud data.
- **Model Evaluation:** Evaluates the models using accuracy, confusion matrix, and plots decision trees.

### Decision Tree Regression

- **Model Creation:** Constructs a Decision Tree Regressor.
- **Accuracy:** Checks the accuracy of the regression model.

---

The code demonstrates how to preprocess data, build decision tree classifiers and regressors, and evaluate their performance on test data. It provides insights into model accuracy, prediction counts, and decision tree visualization.
