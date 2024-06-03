# <p align="center"> :bank: ANK CUSTOMER CHURN PREDICTIONS :chart:
  
# Business Case:  
The high rate of customers leaving banks (churn rate) suggests deficiencies in several areas, including customer experience, operational efficiency, and the competitiveness of products and features. This necessitates a focus on understanding and managing customer churn to improve overall customer satisfaction and achieve sustainable growth. 


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

The dataset is from [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) and stores information for 10,000 bank customers, with each customer represented by a row and 14 features in separate columns. This totals 140,000 data points.

![image](https://github.com/aprilhong/bankchurn/assets/78663820/691081c9-49de-40b0-ae8e-e051f6b94006)

The information includes:
- Customer details: A unique ID, Surname, Gender, and Age.
- Account details: A unique Row Number, Credit Score, Account Balance, and Estimated Salary.
- Banking activity: The number of products the customer has (e.g., checking, savings, loans), whether they have a credit card, and their account activity status (active member or not).
- Outcome: Whether the customer has left the bank (Exited). This is the key piece of information we're trying to understand.

There are two main types of features: numeric and categorical
- 7 numeric features: RowNumber, CustomerId ,CreditScore, Age, Tenure, Balance, EstimatedSalary
- 7 categorical features: Surname, Gender, Geography, NumOfProducts, HasCrCard, IsActiveMember, Exited.

#### Dropping features
  - The **CustomerId** and **Surname** variable has sensitive customer data and should be removed to maintain confidentally. 
  - **Gender** should also be removed as it would be discrimatory to offer promotions based on gender.
  - **RowNumber** can also be removed has it is just a counter.

Now there are 10 features in the dataframe.
![image](https://github.com/aprilhong/bankchurn/assets/78663820/b14a1a4b-c1d3-481c-b642-8d083ed23abe)


#### Descriptive Statistics

![image](https://github.com/aprilhong/bankchurn/assets/78663820/50003524-a5cc-4789-a8dd-9818ac75568f)
- **Credit scores** range widely, from 350 to 850 with an average of 650.
- The typical customer is around **38 years old**, with ages ranging from 18 to 92.
- Customers have been with the bank for an average of **5 years (tenure)**.
- Account **balances** vary greatly, from practically zero up to $250,000.
- Customers' estimated salaries show a broad range, falling between $11.58 and $199,000.

#### Data Cleaning
There are **no missing or duplicated** data but there are **outliers** for **Credit Score** and **Age** features. 

![image](https://github.com/aprilhong/bankchurn/assets/78663820/7bfe3a5e-16a0-4390-b3bb-0b7b3b412905)

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/0cc687b9-1058-41e9-96a1-d83767da8563" width="150" >

- There are 15 customers with scores below 383
- and a sizeable group (359 customers) over the age of 62. 

### Variable Analysis and Visualization

#### `Exited`
Start by checking the class imbalance for Exited since it is a categorical reponse variable.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c0ce1751-efc6-40a0-b4fc-a71ba539e619" width="350" >

- Out of 10,000 customers, a little over 2,000 (20.37%) have churned. This means that the bank retains a bit less than 80% of its customers.
- While a perfect 50-50 split between churning and retained customers is ideal, an 80-20 split is still considered workable for analysis.
- This suggests that there's a good base of loyal customers to build on and target for further growth.

#### `Age`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/d693e2bd-97e4-4650-8ead-9b163d6581d3" width="350" >

- While 30-40 year olds make up the largest age group,
- The highest number of exits (around 700) came from the 40-50 age bracket.
- This suggests a higher churn rate for customers over 40 compared to those under 40.

Let's check the average customer balance across the age groups

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/139d66ca-a226-465a-845a-5607f6aa95ee" width="500" >

- Customers under 90 who remained had an average balance under $80,000.
- Conversely, customers under 90 with balances exceeding $90,000 have exited.
- Do benefits decrease after reaching $90,000 in accumulated balance?

#### `Balance`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/0a424ca4-a2ce-4bc8-b859-be3626e90b49" width="500" >

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

#### `Active Members`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/dbcaf48f-fe6a-4ceb-adaf-db2d6f5a8244" width="350" >

- Active customers churn at a rate of 14.3%.
- Inactive customers churn at a rate of 26.9%.
- This is 12.6 percentage points higher than the churn rate for active customers.
- In other words, inactive customers are 12.6% more likely to churn than active customers.

#### `Number Of Products`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/baaa0c25-6f4f-4df8-b56c-ac43cf80f2a5" width="350" >

- Customers can have up to 4 products 
- The data shows a clear connection between the number of products a customer holds and their likelihood of churning.
- **Most Common, Most Churn:** Over half (50.84%) of customers have only **1 product**, and this group also has the highest number of churned customers (1409). This suggests they might not be finding enough value in the single product to justify staying.
- **Sweet Spot**: Customers with **2 products** (45.9% of the base) have the lowest churn rate (7.6% or 348 customers). This indicates that having a couple of products increases engagement and loyalty.
- **High Risk, High Chur**n: Customers with **3 or 4 products** (a combined 3.26% of the base) have a very high churn rate (82.7% or all churned for 4 products). This suggests these customers might be overwhelmed by too many options or have niche needs not being met.
- These findings suggest that offering the right bundle of products can significantly impact customer retention.
- It might be beneficial to explore why customers with 3 or 4 products churn and tailor product recommendations for those with only 1 product.

#### `Geography`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/c0b788d0-49b0-435c-9f0e-47bc29c116f0" width="350" >

- The customer base comes from three European countries: France, Germany, and Spain. 
- France holds the majority with 5,014 customers, making up over half of the total.
- The remaining customers are distributed relatively evenly between Germany (2,509) and Spain (2,477).
- Germany and France has similar number of customers churns but interestingly, churn rates vary across these regions.
- Germany has the highest churn rate at 32.4%
- France and Spain experience churn rates around 16.2% and 16.7% respectively.

Let's check customer balance for each country to gain additional insight

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/37421f4f-91bf-4afe-95aa-e9937a90c4e1" width="350" >

- Customers who churned in France and Spain took an average balance of $71,000 and $73,000, respectively.
- However, in Germany, churned customers took an average of nearly twice that amount, at $120,000. This suggests that German churned customers are leaving with a significantly higher balance compared to France and Spain.
- It's also worth noting that Germany has the highest churn rate at 32%, compared to France (16.2%) and Spain (16.7%).
- This could indicate that Germany is losing a higher proportion of customers with larger balances.
- Further investigation into the reasons behind churn in Germany might be beneficial to mitigate customer losses and the associated revenue impact.

Let's look at the distribution of customers balance across these countries.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/6a5b9ecf-e134-487b-a312-fade39acc59a" width="350" >

- France and Spain: Around half of the customers in these countries maintain zero balances. Among those with balances (likely the more profitable customers),they take an average of $71,000 - $73,000 with them when they churn.
- Germany: While Germany has a smaller overall customer base, it's customers, either remaining or churned, have much higher average balances of around $120,000. 

- Focus on Germany:  These findings highlight the importance of prioritizing improvements in Germany's customer service or product offerings. By addressing the reasons behind churn in Germany, the bank can potentially retain more high-value customers and mitigate significant revenue losses.

#### `Credit Score`

According to FICO, the credit score rating as categorized as follows
- Very poor: 300 to 579
- Fair: 580 to 669
- Good: 670 to 739
- Very good: 740 to 799
- Excellent: 800 to 850

Hence, I've created a new feature `CreditRating` by grouping the scores to better visualize the data.

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/8f0117e0-128f-4acc-a2cf-09c1f42824e5" width="400" >

- The highest number of churned customers falls within the "Fair" credit rating category, with approximately 2646 customers.
- Customers with "Very Poor" credit scores also show a significant churn rate, with roughly 520 customers leaving the company.
- The number of churned customers drops for customers with higher credit ratings
- Very Good has around 252 churned customers.
- Excellent has the lowest churn with approximately 128 customers leaving.

The churn rate calculated for each rating group are as follows
- Very Poor: 22.0%
- Fair: 20.6%
- Good: 18.6%
- Very Good: 20.6%
- Excellent: 19.5%

- Surprisingly, credit rating does not have a clear correlation with churn rate.
- Although, customers with a Fair credit rating churned the most, it's 20.6% churn rate is very close to all those for other ratings.
- Even the highest credit rating, Excellent, has a churn rate of 19.5%.
- This suggests that customers with good credit scores are just as likely to churn as those with poor credit scores.

#### `Credit Card`

<img src="https://github.com/aprilhong/bankchurn/assets/78663820/48276029-8458-44f0-9c7a-346b2f28e0a1" width="350" >

- The data reveals a surprising finding about customer churn ("Exited") in relation to credit card ownership.
- Although the number of customers with credit cards churned is more than double those without cards, both groups have very similar churn rates.
- Among 7,055 customers who have a credit card, 1,424 churned, resulting in a churn rate of approximately 20.2%.
- Similarly, for 2,495 customers who don't have a credit card, 613 churned, representing a churn rate of around 20.8%.
- It's unexpected that credit card ownership doesn't have a clearer impact on churn.
- While the churn rates are slightly different, the difference is minimal.

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

