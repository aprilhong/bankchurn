# Business Case: European Bank Churn Predictions with XGBoost Machine Learning Model 

The high rate of customers leaving banks (churn rate) suggests deficiencies in several areas, including customer experience, operational efficiency, and the competitiveness of products and features. This necessitates a focus on understanding and managing customer churn to improve overall customer satisfaction and achieve sustainable growth. 

Data used is 

modeling results
T
Data was trained on 3 models (tuned decision tree, random forest and xgboost) 

###  File Descriptions
<details>
<summary>Expand/Collapse</summary>

  - [data](https://github.com/aprilhong/bankchurn/tree/main/data) : folder containing all data files
    - **churn_data.csv**: raw dataset from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
  - [models](https://github.com/aprilhong/bankchurn/tree/main/models) : folder containing all model files
    - **tree_cv_model.pickle, rf_cv_model.pickle**, **xgb_cv_model.pickle** 
    - **model_results_table.csv** : summary table of scoring metrics from all models
    - **xgb_decision_tree.png** : decision tree output from xgb prediction.
  - [requirements.txt](https://github.com/aprilhong/bankchurn/blob/main/requirements.txt) : set up to install all listed packages in the development environment
  - **results_table** : module to create a table from model's evaluation metrics.
</details>
  
# Executive Summary

## 1. Exploratory Data Analysis
Explain what data you used in your analysis, the timeframe of the data, and any data limitations. This is also a good section to add visualizations of your exploratory data analysis.
  1. Data Discovery
  2. Data Cleaning
  3. Variable Analysis and Visualizations
## 2. Feature Engineering
  1. Feature Transformation
## 3. Modeling and Evaluation
This section should detail what models you used and the corresponding evaluation metrics.
### Evaluation Metric
In predicting customer churn, the model can make two types of mistakes:

- **False positive**: The model predicts a customer will leave (churn) but they stay. This might lead to unnecessary efforts to retain the customer.
- **False negative**: The model predicts a customer will stay but they end up leaving. This can be more costly, as the bank misses the chance to intervene.
Since it's more critical to avoid missing churned customers, focusing on recall (catching churners) seems ideal. However, prioritizing recall alone could lead the model to mistakenly predict churn for many customers who wouldn't actually leave. This would result in the bank wasting resources on unnecessary customer retention efforts.

To strike a balance, the F1 score is a better metric to use. It considers both recall and precision (correctly identifying non-churners), giving a more accurate picture of the model's performance. 

### Modeling Approach
The objective of the model is to predict the categorical **Exited** variable; whether a customer will churn or not. Hence we will be training the data on several classification machine learning models and compare their f1 score to determine the champion model. The champion model will then be used to predict on the test data.  

### Model 1: Decision Tree (quick summaries and link to notebook)

### Model 2: Random Forest (quick summaries and link to notebook)
### Model 3: XGBoost (expand)
### Best Model
### Results
     
## Conclusion
In the conclusion section explain the recommendations you have in solving the business problem and highlight any future steps you will take to expand on your project,

