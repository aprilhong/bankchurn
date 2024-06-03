# <p align="center"> :bank: EUROPEAN BANK CHURN PREDICTIONS :chart:
  
# Business Case:  
The high rate of customers leaving banks (churn rate) suggests deficiencies in several areas, including customer experience, operational efficiency, and the competitiveness of products and features. This necessitates a focus on understanding and managing customer churn to improve overall customer satisfaction and achieve sustainable growth. 

Data used is 

modeling results
T
Data was trained on 3 models (tuned decision tree, random forest and xgboost) 

### Table of Content
<details><summary>Expand/Collapse</summary>

1. [File Descriptions](#file-descriptions)
2. [Technologies Used](#technologies-used)
3. [Executive Summary](#executive-summary)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
    1. [Data Discovery](#data-discovery)
    2. [Data Cleaning](#data-cleaning)
    3. [Variable Analysis and Visualization](#variable-analysis-and-visualization)
6. [Feature Engineering](#feature-engineering)
7. [Modeling and Evaluation](#modeling-and-evaluation)
   1. [Modeling Approach](#modeling-approach)
   3. [Evaluation Metric](#evaluation-metric)
   4. [Model 1: Decision Tree](#model-1-decision-tree)
   5. [Model 2: Random Forest](#model-2-random-forest)
   6. [Model 3: XGBoost](#model-3-xgboost)
   7. [Champion Model](#champion-model)
   8. [Evaluation Results](#results)
9. [Conclusion](#conclusion)
</details>

### File Descriptions

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

## Exploratory Data Analysis

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

- There are 15 customers with scores below 383
- and a sizeable group (359 customers) over the age of 62. 

### Variable Analysis and Visualization

#### Exited
Start by checking the class imbalance for Exited since it is a categorical reponse variable.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/c0ce1751-efc6-40a0-b4fc-a71ba539e619)

Out of 10,000 customers, a little over 2,000 (20.37%) have churned. This means that the bank retains a bit less than 80% of its customers.

While a perfect 50-50 split between churning and retained customers is ideal, an 80-20 split is still considered workable for analysis. This suggests that there's a good base of loyal customers to build on and target for further growth.

#### Age
![image](https://github.com/aprilhong/bankchurn/assets/78663820/f8a4a70e-b91d-4ba4-b26d-768f5fd38770)

- While 30-40 year olds make up the largest age group,
- The highest number of exits (around 700) came from the 40-50 age bracket.
- This suggests a higher churn rate for customers over 40 compared to those under 40.

Let's check the average custoemr balance across the age groups
![image](https://github.com/aprilhong/bankchurn/assets/78663820/f5a90003-f99d-457c-b7a4-3d6baac47a7b)

- Customers under 90 who remained had an average balance under $80,000.
- Conversely, customers under 90 with balances exceeding $90,000 have exited.
- Do benefits decrease after reaching $90,000 in accumulated balance?

#### Balance

![image](https://github.com/aprilhong/bankchurn/assets/78663820/086c4fad-ebc3-4df9-8c1e-ad03720b92ff)

- A significant portion (over 3,500 or 35%) of customers have zero balance.
- Interestingly, a quarter (around 500) of those with zero balance have exited.
- Notably, this represents a large portion (around 25%) of all exiting customers (2,037), suggesting a potential link between zero balance and customer churn.

Let's filter out customers with zero balance and plot them against other features.
![image](https://github.com/aprilhong/bankchurn/assets/78663820/78fe69b1-2881-4e13-acca-526da461f0ac)

Here's a breakdown of the 500 customers who exited with zero balance:
- Short Tenure: Roughly 28% (around 140) left within the first two years of opening their account. This suggests they might not have found the value they were looking for early on.
- Millennial Focus: Around 40% (around 200) were between 30 and 40 years old. This age group may have different banking needs or priorities compared to other demographics.
- Limited Engagement: Over 60% (more than 300) only had one product with the bank. This indicates they might not have been fully utilizing the bank's offerings.
- Credit Card Users: More than 60% (more than 300) had a credit card. This doesn't necessarily explain their exit, but it could be a factor to consider.
- Inactive Accounts: Over 60% (more than 300) were not actively using their account. This inactivity could be a reason for the account closure or a sign of dissatisfaction.

#### Active Members

![image](https://github.com/aprilhong/bankchurn/assets/78663820/e8e870b0-6625-4496-8f5c-0466b22acbf3)

- Active customers churn at a rate of 14.3%.
- Inactive customers churn at a rate of 26.9%. This is 12.6 percentage points higher than the churn rate for active customers. In other words, inactive customers are 12.6% more likely to churn than active customers.
- 
#### Num Of Products

![image](https://github.com/aprilhong/bankchurn/assets/78663820/baaa0c25-6f4f-4df8-b56c-ac43cf80f2a5)

- Customers can have up to 4 products 
- The data shows a clear connection between the number of products a customer holds and their likelihood of churning.
- Most Common, Most Churn: Over half (50.84%) of customers have only 1 product, and this group also has the highest number of churned customers (1409). This suggests they might not be finding enough value in the single product to justify staying.
- Sweet Spot: Customers with 2 products (45.9% of the base) have the lowest churn rate (7.6% or 348 customers). This indicates that having a couple of products increases engagement and loyalty.
- High Risk, High Churn: Customers with 3 or 4 products (a combined 3.26% of the base) have a very high churn rate (82.7% or all churned for 4 products). This suggests these customers might be overwhelmed by too many options or have niche needs not being met.
- These findings suggest that offering the right bundle of products can significantly impact customer retention.
- It might be beneficial to explore why customers with 3 or 4 products churn and tailor product recommendations for those with only 1 product.

#### Geography 
 - Germany has the highest churn percentage at ~32%, whereas France and Spain are similar around ~16%
    - In Germany, balance amount for Exited customers averages $120K.
    - Despite having the most customers, France's average customer balance is just half of Germany's ($60K)

 
## Feature Engineering
  1. Feature Transformation
  2. Feature Selection
## Modeling and Evaluation
This section should detail what models you used and the corresponding evaluation metrics.

### Modeling Approach
The objective of the model is to predict the categorical **Exited** variable; whether a customer will churn or not. Hence we will be training the data on several classification machine learning models and compare their f1 score to determine the champion model. The champion model will then be used to predict on the test data. 

### Evaluation Metric
In predicting customer churn, the model can make two types of mistakes:

- **False positive**: The model predicts a customer will leave (churn) but they stay. This might lead to unnecessary efforts to retain the customer.
- **False negative**: The model predicts a customer will stay but they end up leaving. This can be more costly, as the bank misses the chance to intervene.
Since it's more critical to avoid missing churned customers, focusing on recall (catching churners) seems ideal. However, prioritizing recall alone could lead the model to mistakenly predict churn for many customers who wouldn't actually leave. This would result in the bank wasting resources on unnecessary customer retention efforts.

To strike a balance, the F1 score is a better metric to use. It considers both recall and precision (correctly identifying non-churners), giving a more accurate picture of the model's performance. 


### Model 1 Decision Tree 
(quick summaries and link to notebook)

### Model 2 Random Forest 
(quick summaries and link to notebook)
### Model 3 XGBoost 
(expand)
### Champion Model
### Results
## Conclusion
In the conclusion section explain the recommendations you have in solving the business problem and highlight any future steps you will take to expand on your project,

