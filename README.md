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
For our model prediction, the 2 possibles for bad predictions are a false positive and false negative. A false positive is when the model predicts a customer will churn but they did not and a false negative occurs when the model predicts customer will NOT churn but they do. Since the cost of predicting a false negative is higher than that of a false positive, recall would be a good metric to consider. However, using recall only can result in a bias model predicting a majority of customers would churn. In other words, the bank could be offering promotions/discounts to more customers than needed. On the other hand, the f1 score is harmonic mean between recall and precision and would be the best metric to use for our model predictions. 

### Modeling Approach
Reponse vairable is categorical Classifcation models 
evalutae 3 models and choose best one to fit the test data 

### Model 1: Decision Tree (quick summaries and link to notebook)

### Model 2: Random Forest (quick summaries and link to notebook)
### Model 3: XGBoost (expand)
### Best Model
### Results
     
## Conclusion
In the conclusion section explain the recommendations you have in solving the business problem and highlight any future steps you will take to expand on your project,

