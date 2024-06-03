# <p align="center"> :bank: EUROPEAN BANK CHURN PREDICTIONS :chart:
  
# Business Case:  
The high rate of customers leaving banks (churn rate) suggests deficiencies in several areas, including customer experience, operational efficiency, and the competitiveness of products and features. This necessitates a focus on understanding and managing customer churn to improve overall customer satisfaction and achieve sustainable growth. 

Data used is 

modeling results
T
Data was trained on 3 models (tuned decision tree, random forest and xgboost) 

###  File Descriptions

<details><summary>Expand/Collapse</summary>
  
  - [data](https://github.com/aprilhong/bankchurn/tree/main/data) : folder containing all data files
  - **churn_data.csv**: raw dataset from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
  - [models](https://github.com/aprilhong/bankchurn/tree/main/models) : folder containing all model files
    - **tree_cv_model.pickle, rf_cv_model.pickle**, **xgb_cv_model.pickle** 
    - **model_results_table.csv** : summary table of scoring metrics from all models
    - **xgb_decision_tree.png** : decision tree output from xgb prediction.
  - [requirements.txt](https://github.com/aprilhong/bankchurn/blob/main/requirements.txt) : set up to install all listed packages in the development environment
  - **results_table** : module to create a table from model's evaluation metrics.
</details>

### Technologies Used

<details> <Summary>Expand/Collapse</summary>
  
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn
</details>

# Executive Summary

## 1. Exploratory Data Analysis

### Data Discovery
The dataset is from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) and stores information for 10,000 bank customers, with each customer represented by a row and 14 features in separate columns. This totals 140,000 data points.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/691081c9-49de-40b0-ae8e-e051f6b94006)

The information includes:
- Customer details: A unique ID, Surname, Gender, and Age.
- Account details: A unique Row Number, Credit Score, Account Balance, and Estimated Salary.
- Banking activity: The number of products the customer has (e.g., checking, savings, loans), whether they have a credit card, and their account activity status (active member or not).
- Outcome: Whether the customer has left the bank (Exited). This is the key piece of information we're trying to understand.

There are two main types of features:
- 7 numeric features: RowNumber, CustomerId ,CreditScore, Age, Tenure, Balance, EstimatedSalary
- 7 categorical features: Surname, Gender, Geography, NumOfProducts, HasCrCard, IsActiveMember, Exited.

### Dropping features
  - The **CustomerId** and **Surname** variable has sensitive customer data and should be removed to maintain confidentally. 
  - **Gender** should also be removed as it would be discrimatory to offer promotions based on gender.
  - **RowNumber** can also be removed has it is just a counter. 

### Descriptive Statistics

![image](https://github.com/aprilhong/bankchurn/assets/78663820/50003524-a5cc-4789-a8dd-9818ac75568f)
- **Credit scores** range widely, from 350 to 850 with an average of 650.
- The typical customer is around **38 years old**, with ages ranging from 18 to 92.
- Customers have been with the bank for an average of **5 years (tenure)**.
- Account **balances** vary greatly, from practically zero up to $250,000.
- Customers' estimated salaries show a broad range, falling between $11.58 and $199,000.

### Data Cleaning
There are **no missing or duplicated** data but there are **outliers** for **Credit Score** and **Age** features. 

![image](https://github.com/aprilhong/bankchurn/assets/78663820/b959b1e2-51f0-4043-b4e8-136d19859f41)

Calculate number of outliers per feature

![image](https://github.com/aprilhong/bankchurn/assets/78663820/efcf450a-67e6-4be2-814d-2075d5d88047)


- There are 15 customers with credit scores under 383. 
- There are 359 customers over the age of 62.



### Variable Analysis and Visualizations

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

