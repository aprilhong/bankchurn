# <p align="center"> :bank: Predicting Customer Churn at UK Bank:chart:
  
## Problem Statement
Banks face a significant challenge in retaining customers, as customer churn can lead to substantial financial losses and hinder business growth. Identifying customers at risk of churning is crucial for banks to implement proactive retention strategies and maintain customer loyalty. Churn can result in lost revenue, increased acquisition costs, damaged reputation, reduced market share, and operational inefficiencies.

## Objective
To develop a robust machine learning model capable of accurately predicting customer churn within a banking context. This model will serve as a valuable tool for marketing analysts to identify at-risk customers, enabling them to design targeted retention campaigns and optimize marketing budgets for maximum impact.

**Methodology**:

1. Exploratory Data Analysis to reveal insights:
    - Higher churn rate for customers over 40
    - Inactive customers are 12.6% more likely to churn than active customers.
    - Customers with 2 products have the lowest churn rate
    - Germany has the highest churn rate at 32%, compared to France (16.2%) and Spain (16.7%).
2. Train various classification models (decision tree, random forest, XGBoost) on historical customer data.
3. Evaluate model performance using the F1 score, which balances identifying true churners (recall) and avoiding false positives.
4. Implement the XGBoost model on test data yielded good F1 score, (0.77) indicating a strong balance between precision and recall. 

**Recommendations**: 
- Conduct a deeper analysis of churn in Germany, particularly among high-balance customers, to identify and address specific reasons for their departure.
- Investigate the 25% of churned customers with zero balance. Understand why they weren't actively using our services and develop strategies to re-engage them.
- Focus on the most important features, like income, credit score, account balance, and age, might reveal additional insights to enhance the model's performance.

**Conclusion**: Investing in a customer churn prediction model empowers data-driven decision making, leading to improved customer retention, enhanced customer experience, and ultimately, sustainable business growth.

### Table of Content
<details><summary>Expand/Collapse</summary>

1. [File Descriptions](#file-descriptions)
2. [Technologies Used](#technologies-used)
3. [Executive Summary](#executive-summary)
    1. [Exploratory Data Analysis](#exploratory-data-analysis)
       - [Data Cleaning](#data-cleaning)
       - [Variable Analysis and Visualization](#variable-analysis-and-visualization)
    2. [Feature Engineering](#feature-engineering)
    3. [Modeling](#modeling)
       - [Evaluation Metric](#evaluation-metric)
       - [Model 1: Decision Tree](#model-1-decision-tree)
       - [Model 2: Random Forest](#model-2-random-forest)
       - [Model 3: XGBoost](#model-3-xgboost)
       - [Champion Model](#champion-model)
    4. [Evaluation](#evaluation)
       - [Reccomendations](#recommendations)
   
</details>

### File Descriptions

<details><summary>Expand/Collapse</summary>
  
  - [data](https://github.com/aprilhong/bankchurn/tree/main/data) : folder containing all data files
  - **churn_data.csv**: raw dataset from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
  - [models](https://github.com/aprilhong/bankchurn/tree/main/models) : folder containing all model files
    - **tree_cv_model.pickle, rf_cv_model.pickle**, **xgb_cv_model.pickle** 
    - **model_results_table.csv** : summary table of scoring metrics from all models
    - **xgb_decision_tree.png** : decision tree output from xgb prediction on test data. 
  - [bankchurn.ipynb](https://github.com/aprilhong/bankchurn/blob/main/bankchurn.ipynb) - notebook will full analysis
  - [requirements.txt](https://github.com/aprilhong/bankchurn/blob/main/requirements.txt) : set up to install all listed packages in the development environment
  - [results_table.py](https://github.com/aprilhong/bankchurn/blob/main/results_table.py) : module to create a table from model's evaluation metrics.
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
<details><summary>Expand/Collapse</summary>
The dataset is from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) and stores information for 10,000 bank customers, with each customer represented by a row and 14 features in separate columns. This totals 140,000 data points.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/691081c9-49de-40b0-ae8e-e051f6b94006)

The information includes:
- **Customer details**: A unique ID, Surname, Gender, and Age.
- **Account details**: A unique Row Number, Credit Score, Account Balance, and Estimated Salary.
- **Banking activity**: The number of products the customer has (e.g., checking, savings, loans), whether they have a credit card, and their account activity status (active member or not).
- **Outcome**: Whether the customer has left the bank (Exited). This is the key piece of information we're trying to understand.

There are two main types of features: numeric and categorical
- 7 numeric features: RowNumber, CustomerId ,CreditScore, Age, Tenure, Balance, EstimatedSalary
- 7 categorical features: Surname, Gender, Geography, NumOfProducts, HasCrCard, IsActiveMember, Exited.

### Dropping features
  - The **CustomerId** and **Surname** variable has sensitive customer data and should be removed to maintain confidentally. 
  - **Gender** should also be removed as it would be discrimatory to offer promotions based on gender.
  - **RowNumber** can also be removed has it is just a counter.

The new shape after dropping these features are (10000,10). 

![image](https://github.com/aprilhong/bankchurn/assets/78663820/b14a1a4b-c1d3-481c-b642-8d083ed23abe)


### Descriptive Statistics

![image](https://github.com/aprilhong/bankchurn/assets/78663820/50003524-a5cc-4789-a8dd-9818ac75568f)
- **Credit scores** range from 350 to 850 with an average of 650.
- The typical customer is around **38 years old**, with ages ranging from 18 to 92.
- Customers have been with the bank for an average of **5 years (tenure)**.
- Account **balances** vary greatly, from zero up to $250,000.
- Customers' estimated salaries show a broad range, falling between $11.58 and $199,000.

### Data Cleaning
There are **no missing or duplicated** data but there are **outliers** for **Credit Score** and **Age** features. 

![image](https://github.com/aprilhong/bankchurn/assets/78663820/7bfe3a5e-16a0-4390-b3bb-0b7b3b412905)

- There are about 15 customers with credit scores below 383
- and a sizeable group (359 customers) over the age of 62. 

### Variable Analysis and Visualization

#### `Exited`
Start by checking the class imbalance for Exited since it is a categorical reponse variable.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c0ce1751-efc6-40a0-b4fc-a71ba539e619" width="350" >


- Out of 10,000 customers, a little over 2,000 (20.37%) have churned. This means that the bank retains a bit less than 80% of its customers.
- While a perfect 50-50 split between churning and retained customers is ideal, an 80-20 split is still considered workable for the analysis.
- This suggests that there's a good base of loyal customers to build on and target for further growth.

#### `Age`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/d693e2bd-97e4-4650-8ead-9b163d6581d3" width="400" >

- While **30-40 year olds** make up the largest age group,
- The highest number of exits (around 700) came from the **40-50 age** bracket.
- This suggests a higher churn rate for customers over 40 compared to those under 40.

Let's check the average customer balance across the age groups

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/139d66ca-a226-465a-845a-5607f6aa95ee" width="500" >

- Customers under 90 who remained had an average balance under $80,000.
- Conversely, customers under 90 with balances **exceeding $90,000 have exited**.
- Do benefits decrease after reaching $90,000 in accumulated balance?

#### `Balance`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/0a424ca4-a2ce-4bc8-b859-be3626e90b49" width="500" >

- A significant portion (over 3,500 or 35%) of customers have **zero balance**.
- Interestingly, a **quarter** (around 500) of those with zero balance **have exited**.
- Notably, this represents a large portion (around 25%) of all exiting customers (2,037), suggesting a potential link between zero balance and customer churn.

Let's filter out customers with zero balance and plot them against other features.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/78fe69b1-2881-4e13-acca-526da461f0ac)

Here's a breakdown of the 500 customers who exited with zero balance:
- **Short Tenure**: Roughly 28% left within the first two years of opening their account. This suggests they might not have found the value they were looking for early on.
- **Millennial Focus**: Around 40%  were between 30 and 40 years old. This age group may have different banking needs or priorities compared to other demographics.
- **Limited Engagement**: Over 60%  only had one product with the bank. This indicates they might not have been fully utilizing the bank's offerings.
- **Credit Card Users**: More than 60% had a credit card. This doesn't necessarily explain their exit, but it could be a factor to consider.
- **Inactive Accounts**: Over 60%  were not actively using their account. This inactivity could be a reason for the account closure or a sign of dissatisfaction.

#### `Active Members`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/dbcaf48f-fe6a-4ceb-adaf-db2d6f5a8244" width="350" >

- **Active customers** churn at a rate of 14.3%.
- **Inactive customers** churn at a rate of 26.9%.
- This is 12.6 percentage points higher than the churn rate for active customers.
- In other words, inactive customers are 12.6% more likely to churn than active customers.

#### `Number Of Products`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/baaa0c25-6f4f-4df8-b56c-ac43cf80f2a5" width="350" >

- Customers can have up to 4 products 
- The data shows a clear connection between the number of products a customer holds and their likelihood of churning.
- Over half of customers have only **1 product**, and this group also has the highest number of churned customers (1409). This suggests they might not be finding enough value in the single product to justify staying.
- Customers with **2 products** (45.9% of the base) have the lowest churn rate (7.6% or 348 customers). This indicates that having a couple of products increases engagement and loyalty.
- Customers with **3 or 4 products** (a combined 3.26% of the base) have a very high churn rate (82.7% or all churned for 4 products). This suggests these customers might be overwhelmed by too many options or have niche needs not being met.
- These findings suggest that offering the right bundle of products can significantly impact customer retention.
- It might be beneficial to explore why customers with 3 or 4 products churn and tailor product recommendations for those with only 1 product.

#### `Geography`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c0b788d0-49b0-435c-9f0e-47bc29c116f0" width="350" >

- The customer base comes from three European countries: **France, Germany, and Spain**. 
- France holds the majority with 5,014 customers, making up over half of the total.
- The remaining customers are distributed relatively evenly between Germany (2,509) and Spain (2,477).
- Germany and France has **similar number of customers churns** but interestingly, churn rates vary across these regions.
- Germany has the **highest churn rate** at 32.4%
- France and Spain experience churn rates around 16.2% and 16.7% respectively.

Let's check customer balance for each country to gain additional insight

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/37421f4f-91bf-4afe-95aa-e9937a90c4e1" width="350" >

- Customers who churned in France and Spain took an average balance of **$71,000 and $73,000**, respectively.
- However, in Germany, churned customers took an average of nearly twice that amount, at **$120,000**. This suggests that German churned customers are leaving with a significantly higher balance compared to France and Spain.
- It's also worth noting that Germany has the highest churn rate at 32%, compared to France (16.2%) and Spain (16.7%).
- This could indicate that Germany is losing a higher proportion of customers with larger balances.
- Further investigation into the reasons behind churn in Germany might be beneficial to mitigate customer losses and the associated revenue impact.

Let's look at the distribution of customers balance across these countries.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/6a5b9ecf-e134-487b-a312-fade39acc59a" width="350" >

- France and Spain: Around **half of the customers** in these countries maintain **zero balances**. Among those with balances (likely the more profitable customers),they take an average of $71,000 - $73,000 with them when they churn.
- Germany: While Germany has a smaller overall customer base, it's customers, either remaining or churned, have much higher average balances of around $120,000. 

- **Focus on Germany**:  These findings highlight the importance of prioritizing improvements in Germany's customer service or product offerings. By addressing the reasons behind churn in Germany, the bank can potentially retain more high-value customers and mitigate significant revenue losses.

#### `Credit Score`

According to FICO, the credit score rating as categorized as follows
- Very poor: 300 to 579
- Fair: 580 to 669
- Good: 670 to 739
- Very good: 740 to 799
- Excellent: 800 to 850

Hence, I've created a new feature `CreditRating` by grouping the scores to better visualize the data.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/8f0117e0-128f-4acc-a2cf-09c1f42824e5" width="400" >

- The highest number of churned customers falls within the **Fair** credit rating category, with approximately 2646 customers.
- Customers with "Very Poor" credit scores also show a significant churn rate, with roughly 520 customers leaving the company.
- The number of churned customers drops for customers with higher credit ratings
  - **Very Good** has around 252 churned customers.
  - **Excellent** has the **lowest number of churned** customers at ~128

The churn rate calculated for each rating group are as follows
  - **Very Poor: 22.0%**
  - Fair: 20.6%
  - Good: 18.6%
  - Very Good: 20.6%
  - **Excellent: 19.5%**

Surprisingly, credit rating does not have a clear correlation with churn rate.
- Although, customers with a Fair credit rating churned the most, it's 20.6% churn rate is very close to all those for other ratings.
- Even the highest credit rating, Excellent, has a churn rate of 19.5%.
- This suggests that customers with good credit scores are just as likely to churn as those with poor credit scores.

#### `Credit Card`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/48276029-8458-44f0-9c7a-346b2f28e0a1" width="350" >

- Although the number of churned customers with credit cards is more than double those without cards, both groups have very similar churn rates.
- Among customers **with a credit card**, 1,424 out of 7055 have churned, resulting in a **20.2% churn rate**
- Similarly, for customers **without a credit card**, 613 out of 2,495 haved churned, representing a **20.8% churn rate**
- While the churn rates are slightly different, the **difference is minimal**

#### `Estimated Salary`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/06d08ef2-52d0-4d98-a927-6c40f4323e1a" width="500" >

- The salary range seems to be evenly spread across the customer base ("uniform distribution"). This means there aren't any specific salary brackets with a higher concentration of customers.
- Regardless of salary range, approximately 25% of customers churn (around 250 customers). This suggests that churn might be driven by factors other than salary.

#### `Tenure`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/1f458da6-5503-4fe3-bb66-4915d9f1e532" width="500" >

- Tenure seems to be evenly distributed across the customer base ("uniform distribution"). Interestingly, the number of customer churns stays relatively consistent at around 200 people every year, except for the first and tenth years.
- Unlike other years, both year 1 and year 10 have a lower churn rate, with only around 100 customers churning in each of these years.

To understand the reasons behind the lower churn rates in year 1 and year 10, it would be beneficial to:
- Year 1 Retention: Investigate what the bank might be doing well to retain customers in the first year. It could be strong onboarding processes, competitive introductory offers, or meeting the initial needs of new customers effectively.
- Year 10 Loyalty: Explore why customers reach a decade with the bank and why they churn at a lower rate then. Possible explanations include established loyalty programs, inertia (less likely to switch after a long tenure), or the bank effectively catering to the needs of long-term customers.


## Feature Engineering

### Feature Transformation
The Geography feature needs to be encoded first.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/93b42fb3-8276-4b8d-af36-ce501b9b9be5)


### Correlation Matrix
<img src="https://github.com/aprilhong/bankchurn/assets/78663820/60c640c0-53af-4fb8-8065-2a35905b17f8" width="600" >

Positive Correlations
- 0.37: Geography_Germany & Balance
- 0.32: Exited & Age
- 0.17: Exited & Geography_Germany
- 0.11: Exited & Balance

Negative Correlations
- -0.32: NumofProducts & Balance
- -0.16: Exited & IsActiveMember
- -0.13: Exited & NumOfProducts
- -0.10: Exited & Geography_France

### Features Selected
- Target: Exited
- Predictive: Age, CreditScore, HasCrCard, IsActiveMember, Geography_Germany, Geography_France, Geography_Spain, Balance, NumOfProducts, EstimatedSalary,
</details>

## Modeling
<details><summary>Expand/Collapse</summary>

The objective is to build a model that predict whether a customer is likely to churn. This will be done by training various classification models, including decision tree, random forest, and xgboost, on our existing data. The model with the best performance, measured by F1 score, will be chosen as the winner. This winning model will then be used to predict churn on new, unseen data.

### Evaluation Metric
In predicting customer churn, the model can make two types of mistakes:

- **False positive**: The model predicts a customer will leave (churn) but they stay. This might lead to unnecessary efforts to retain the customer.
- **False negative**: The model predicts a customer will stay but they end up leaving. This can be more costly, as the bank misses the chance to intervene.
Since it's more critical to avoid missing churned customers, focusing on recall (catching churners) seems ideal. However, prioritizing recall alone could lead the model to mistakenly predict churn for many customers who wouldn't actually leave. This would result in the bank wasting resources on unnecessary customer retention efforts.

To strike a balance, the **F1 score** is a better metric to use. It considers both recall and precision (correctly identifying non-churners), giving a more accurate picture of the model's performance. 



### Split Data into training and testing dataset
Note on outliers: Since the models used are variations of the decision tree algorithms which is able to handle outliers easily. The outliers will not be removed from the dataset. 

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/5ffb5a79-621b-432e-af94-d728d5233487" width="600" >

Both the X and y variables are separated and the data is split 75% for training and 25% for testing. To maintain the class imbalance in the training and testing datasets, stratify was set equal to y. A random state of 38 was selected for reproducibiliy purposes. 

### Model 1 Decision Tree 
The decision tree model serves as the baseline to compare the performance of more complex models like random forest and xgboost. The F1 score of the decision tree becomes a baseline that the other models need to surpass to be considered an improvement.

#### The scores for the baseline model are as follows

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/4d759b33-aa62-4e1f-b631-418e22d67506" width="500" >

- Precision: Of all positive predictions, 46.7% prediction are true positive.
- Recall: Of all real positive cases, 48.3% are predicted positive.
- **F1 Score**: the test set's harmonic mean is **47.5%**
- Accuracy: Of all cases in test set, 78.3% are predicted true positive or true negative.

#### Confusion Matrix

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/4e053853-cdde-436b-9fd0-9f651d81ae24" width="350" >

- True Positives of (246): These are the customers the model correctly identified as churning (Exited = 1). 
- False Negatives (263): These are the customers the model incorrectly classified as not churning (Exited = 0) but actually churned.
- True Negatives (1711): These are the customers the model correctly identified as not churning (Exited = 0)
- False Positives (280): These are the customers the model incorrectly classified as churning (Exited = 1) but did not churn.

#### Feature Importance
When building the decision tree, the algorithm considers all features and chooses the one that results in the biggest decrease in Gini impurity after splitting the data. This decrease in impurity reflects how well that feature separates the data into classes relevant to the target variable (Exited).

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/1c920e88-e27c-4c89-8d1c-f3e818136af0" width="500" >

The top 3 features based on gini impurity are **Age, Estimated Salary, and Balance**. 
- Age is responsible for 22% of overall reduction of gini impurity in the model
- EstimatedSalary is responsible for 17% of overall reduction of gini impurity in the model
- Balance is responsible for 17% of overall reduction of gini impurity in the model

#### Baseline Model Evaluation Score
An F1 score of **0.475** is considered **poor** in machine learning classification tasks, hence, the next task is to tune the decision tree.

### Tuned Decision Tree
After fitting the training data to the GridSearch CV object, the best parameters were 
- max_depth: 10,
- min_samples_lea: 15
- min_samples_split: 2

#### Tuned Decision Tree Model Evaluation Score

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/8e6f36f4-5f3a-4fd5-b97b-1daf20ad3933" width="500" >

Tuning the decision tree increased F1 from 47% to 56%. Which is an improvement but still not great. Let's consider the random forest model next.

### Model 2 Random Forest 
This model builds multiple decision trees and aggregates their predictions, often leading to better performance and reduced overfitting compared to a single decision tree.

After fitting the training data to the GridSearch CV object, the best parameters were 
- max_depth: None
- max_features: 4
- min_samples_leaf: 3
- min_samples_split: 2
- n_estimators: 150

#### Model Evaluation Score

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c6d9525c-a936-4b69-9df6-e34003418ce5" width="500" >

The random forest model obtained an F1 score of 0.605, which is the highest thus far. But maybe we can do better, let's check out XGBoost.


### Model 3 XGBoost 
XGBoost is another powerful gradient boosting algorithm that might be effective for churn prediction. It can handle complex data relationships and potentially improve F1 score.

After fitting the training data to the GridSearch CV object, the best parameters were 
-learning_rate: 0.2
- max_depth: 6
- min_child_weight: 2
- n_estimators: 125

#### Model Evaluation Score

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/56443e97-1ddc-4384-82ed-746602b5b8c7" width="500" >

The F1 score of 0.606 from this model is only a minimal improvement to the 0.605 from the random forest model. 

### Champion Model
Out of the 3 models **XGBoost** has the highest F1 score; therefore, it will be our champion model to predict on the test data.

#### Use XGBoost Classifer to predict on the test data
<img src="https://github.com/aprilhong/bankchurn/assets/78663820/1f0cfad8-ffa3-47da-9f57-3c723cd8f5d9" width="400">


#### Confusion Matrix
<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c955c377-3b2e-4f4f-8cf8-8f4493e3ac48" width="350">

- Of the 2500 test samples, 509 customers left and the model correctly predicted 345 of the customers
- When the model makes an error, it's typically Type II error giving a false negative, which fails to predict that customer will leave.

#### Feature Importance
<img src="https://github.com/aprilhong/bankchurn/assets/78663820/edda5343-85b4-4701-9523-f4f53da0fe0d" width="400">

From the model, Estimated Salary, CreditScore, Balance, and Age are the the most importance features for predicting customer churn.
</details>

## Evaluation
<details><summary>Expand/Collapse</summary>
<img src="https://github.com/aprilhong/bankchurn/assets/78663820/f388a624-60d6-4e25-bb74-11b3d0a976a9" width="400">

- The F1 scores from all models are ranked in descending order and the the XGBoost score on the test data ranks the highest at 0.77
- An F1 score of 0.77 is considered good in machine learning classification tasks. This indicates that our model is performing well at balancing precision and recall when predicting churn (Exited variable) for the customers.
- Compared to the previous F1 scores of 0.475 and 0.56, this is a significant improvement. It suggests that tuning the decision tree and exploring other algorithms/features, have been effective.

#### Recommendations

From the data analysis, Germany is losing a higher proportion of customers with larger balances. Further investigation into the reasons behind churn in Germany might be beneficial to mitigate customer losses and the associated revenue impact.

A quarter of lost customers had no account balance, suggesting they might not have been actively using our services. The bank can explore following actions. 
  - Bundled Packages: Create package deals that offer multiple products at a discounted rate, encouraging them to explore different services.
  - Re-Engagement Campaigns: Reach out to inactive accounts with targeted email or mobile notifications. Offer incentives for re-activation, like account bonuses or discounts on financial products.
  - Identify Reasons for Inactivity: Conduct surveys or polls to understand why accounts become inactive. This can help address underlying issues and improve overall customer experience.
  - Targeted Onboarding: Millennials value personalization. Develop onboarding experiences that cater to their age group's needs. Offer financial literacy resources, budgeting tools, and goal-setting features to show immediate value.

### Future Improvements

We built a model using most features. To potentially improve its F1 score, we can try simplifying it by removing less important features. We can also explore creating new features from the existing ones, like combining income and age into a financial maturity score. Focusing on the most important features, like income, credit score, account balance, and age, might reveal additional insights to enhance the model's performance.
</details>


