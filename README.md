# Customer Churn Prediction and Retention Strategy

## Project Title
**Subscription Account Attrition Modeling and Risk Stratification Engine via Supervised Classification Machine Learning**

---

## Business Problem
In subscription-based businesses, customer churn directly limits long-term growth and impacts overall profitability. Because acquiring new customers is significantly more expensive than keeping existing accounts active, managing attrition is a top priority for corporate survival.

The objective of this predictive initiative is to build a machine learning pipeline that identifies at-risk accounts before they leave. This enables the company to deploy targeted, proactive retention campaigns and optimize marketing spend.

In customer churn modeling, **False Negatives are far more costly than False Positives**. Missing a truly at-risk customer (False Negative) results in permanent account loss and a direct hit to recurring revenue. On the other hand, misidentifying a loyal customer as at-risk (False Positive) only costs the minor price of a proactive discount or outreach email.

---

## Dataset Description

### Data Structural Profile
The asset comprises historical enterprise CRM customer ledger records, capturing service settings, billing balances, and historical account status tags. This is a classic **supervised learning classification problem** because each row contains an explicit, known historical target label (`Churn`).

### Feature Matrix Definitions
* **CustomerID:** Unique identification text sequence indexing individual accounts.
* **Gender:** The demographic gender profile assigned to the profile.
* **SeniorCitizen:** Binary flag indicator mapping whether the account holder is a senior citizen ($1$) or not ($0$).
* **Partner / Dependents:** Indicators tracking the family demographic structure of the customer.
* **Tenure:** The continuous total number of operational months the customer has spent with the enterprise.
* **PhoneService / MultipleLines:** Service flags logging voice communication selections.
* **InternetService:** Alphanumeric indicator tracking digital data connection types (e.g., DSL, Fiber Optic, No).
* **OnlineSecurity / OnlineBackup / DeviceProtection / TechSupport:** Supplementary value-added infrastructure support flags.
* **Contract:** The explicit customer service model agreement length (e.g., Month-to-month, One year, Two year).
* **PaperlessBilling:** Structural indicator checking if invoicing is handled digitally.
* **PaymentMethod:** The transaction method configuration chosen by the account holder.
* **MonthlyCharges:** The monthly service fee amount billed to the account.
* **TotalCharges:** The cumulative lifetime financial spend of the account.
* **Churn:** The categorical target vector recording whether the customer canceled their service contract ('Yes') or remained active ('No').

---

## Data Cleaning and Preprocessing Summary
To transform the raw records into a machine-learning-ready feature matrix, we executed the following pipeline:

1.  **Imputation Loop:** We filled the 31 missing fields in the `TotalCharges` vector using the dataset's median value. This prevents row loss while preserving the feature's overall distribution.
2.  **Feature Filtering:** We removed the `CustomerID` vector from the modeling data because unique strings do not carry predictive signal and can cause model overfitting.
3.  **Categorical Mapping Vector:** We mapped the categorical `Churn` target variable directly into binary integer formats ($\text{Yes} = 1, \text{No} = 0$).
4.  **One-Hot Matrix Expansion:** We converted multi-class categorical string inputs into binary indicator flags using one-hot encoding, dropping the first category to avoid multicollinearity.
5.  **Stratified Data Splitting:** We split the data into a 75/25 train/test split. We used stratified splitting to ensure the train and test sets have the exact same ratio of churned to retained customers.
6.  **Feature Normalization Scaling:** We standardized continuous numeric columns (`Tenure`, `MonthlyCharges`, `TotalCharges`) to a uniform scale with zero mean and unit variance, ensuring all features contribute equally to the distance calculations.

---

## EDA Insights

* **Baseline Loss Matrix:** The enterprise base has a baseline churn rate of **35.8%**, indicating a clear need for targeted retention strategies.
* **Contract Risks:** Customers on month-to-month contracts exhibit an incredibly high risk of churn, while multi-year contracts provide highly reliable retention.
* **Early-Stage Vulnerability:** Attrition peaks sharply during the first 10 months of an account lifecycle, revealing a critical onboarding window where early customer support is essential.
* **Pricing Pressure:** Churning accounts show a higher median monthly charge than retained ones, indicating that high bills are a primary driver of customer departures.

---

## Models Used
We built and evaluated two distinct classification algorithms:
1.  **Logistic Regression Classifier:** Serves as our parametric linear baseline model.
2.  **Decision Tree Classifier (Max Depth = 5):** Serves as our non-parametric non-linear baseline model.

---

## Model Evaluation Results

Our evaluation on the validation dataset produced the following performance metrics:

### 1. Parametric Logistic Regression Pipeline
* **Accuracy:** 73.89%
* **Precision:** 66.05%
* **Recall:** **55.81%**
* **F1-Score:** 60.50%

### 2. Non-Parametric Decision Tree Pipeline
* **Accuracy:** 72.50%
* **Precision:** 69.74%
* **Recall:** 41.09%
* **F1-Score:** 51.71%

---

## Confusion Matrix Summary (Logistic Regression)
* **True Negatives (TN):** 194 (Retained customers accurately identified as safe).
* **True Positives (TP):** 72 (At-risk customers accurately caught before leaving).
* **False Positives (FP):** 37 (Safe customers flagged as at-risk).
* **False Negatives (FN):** 57 (At-risk customers missed by the model).

---

## Final Model Selection
**Logistic Regression** was chosen as the primary production engine. 

While both models achieved similar overall accuracy, Logistic Regression delivered a significantly higher **Recall score (55.81% vs 41.09%)**. Because missing an at-risk customer carries a high business cost, maximizing recall is the most effective approach for reducing customer churn and protecting recurring revenue.

---

## Retention Strategy Recommendations

### 1. Contract Upgrade Incentives for Month-to-Month Accounts
* **Insight:** Month-to-month contract configurations are the strongest predictor of customer churn.
* **Actionable Recommendation:** *Customers on Month-to-Month contracts exhibit elevated churn risks. The company should launch targeted email campaigns offering a $10 monthly bill credit for 6 months if the customer upgrades to a 1-year or 2-year contract.*

### 2. High-Bill Price Interventions
* **Insight:** Churning accounts are heavily concentrated among high-spending profiles with elevated monthly bills.
* **Actionable Recommendation:** *High monthly charges are strongly correlated with customer attrition. The system should automatically flag accounts when bills cross the $85/month threshold and offer them data bundle optimizations or value-added services to increase satisfaction without cutting prices.*

### 3. Lifecycle Onboarding and Early Care
* **Insight:** Accounts are highly vulnerable during their first 10 months of tenure.
* **Actionable Recommendation:** *Customer attrition peaks sharply during the initial 10 months of tenure. The company should launch a dedicated onboarding program that checks in via automated text or phone support at month 1, 3, and 6 to resolve service friction before it leads to churn.*

### 4. Automated Payment Adjustments
* **Insight:** Electronic check users show significantly higher churn rates due to transaction failures or payment hassle.
* **Actionable Recommendation:** *Accounts utilizing Electronic Checks show high churn correlations. The company should offer a one-time $5 bill credit if these users switch to auto-pay via credit card or direct bank transfer, locking in frictionless billing.*

---

## How to Run the Project

1. Clone this repository down to your computer.
2. Verify that your data file is in the root directory named exactly: `part_3_customer_churn_prediction.csv`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
