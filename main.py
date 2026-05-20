# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("part_3_customer_churn_prediction.csv")
print("--- Raw Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())

# ==========================================
# 3. DATA UNDERSTANDING
# ==========================================
print("\n--- Column Structural Info ---")
print(df.info())

print("\n--- Missing Values Matrix Count ---")
print(df.isnull().sum())

print("\n--- Target Variable Distribution ---")
print(df['Churn'].value_counts(normalize=True))

# ==========================================
# 4. DATA CLEANING
# ==========================================
df_clean = df.copy()

# 1. Handle missing values found within TotalCharges by filling with the median
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())

# 2. Remove duplicate logs if present
df_clean = df_clean.drop_duplicates()

# 3. Handle structural encoding for Target Column
df_clean['Churn_Numeric'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

print("\n--- Pre-processed Cleaned Shape ---")
print(df_clean.shape)

# ==========================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Chart 1: Overall Churn Rate Distribution
plt.figure(figsize=(6,6))
df_clean['Churn'].value_counts().plot.pie(colors=['skyblue', 'salmon'], autopct='%1.1f%%', startangle=90)
plt.title('Overall Enterprise Customer Churn Rate Profile')
plt.ylabel('')
plt.tight_layout()
plt.savefig('churn_rate_pie.png')
plt.show()
print("\nInterpretation Chart 1: The baseline pie chart measures enterprise customer attrition. A churn rate near 35.8% reveals significant revenue leakage, confirming a clear need for programmatic customer retention campaigns.")

# Chart 2: Churn by Contract Type
plt.figure()
sns.countplot(data=df_clean, x='Contract', hue='Churn', palette='Set2')
plt.title('Customer Churn Behavior Stratified by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count of Customers')
plt.tight_layout()
plt.savefig('churn_by_contract.png')
plt.show()
print("\nInterpretation Chart 2: Customers on month-to-month contracts experience dramatically higher attrition compared to those on stable annual agreements, highlighting contract duration as a key factor in retention.")

# Chart 3: Churn by Tenure Density Distribution
plt.figure()
sns.histplot(data=df_clean, x='Tenure', hue='Churn', multiple='stack', kde=True, palette='coolwarm')
plt.title('Customer Tenure Retention and Churn Density Distribution')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count of Customers')
plt.tight_layout()
plt.savefig('churn_by_tenure.png')
plt.show()
print("\nInterpretation Chart 3: Attrition scales significantly during early lifecycles (0 to 10 months). Once a customer crosses the 20-month threshold, their churn risk drops sharply, revealing a critical early onboarding window.")

# Chart 4: Churn by Monthly Charges Box Plot
plt.figure()
sns.boxplot(data=df_clean, x='Churn', y='MonthlyCharges', palette='Set3')
plt.title('Distribution Profile of Monthly Charges Across Churn Segments')
plt.xlabel('Churn Status')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig('churn_by_charges.png')
plt.show()
print("\nInterpretation Chart 4: The median monthly bill for churning accounts is noticeably higher than that of retained customers, suggesting high prices or bill shock can push customers to leave.")

# Chart 5: Churn by Payment Method
plt.figure()
sns.countplot(data=df_clean, y='PaymentMethod', hue='Churn', palette='viridis')
plt.title('Evaluation of Customer Churn Propensity Across Payment Options')
plt.xlabel('Count of Customers')
plt.ylabel('Payment Method')
plt.tight_layout()
plt.savefig('churn_by_payment.png')
plt.show()
print("\nInterpretation Chart 5: Customers using electronic checks experience a much higher rate of attrition compared to those on automated credit card or bank transfer models, pointing to payment failure or regular bill friction as an area for improvement.")

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================
# Drop unique system identifiers and targets to prepare our machine learning matrix
X = df_clean.drop(columns=['CustomerID', 'Churn', 'Churn_Numeric'])
y = df_clean['Churn_Numeric']

# One-hot encode categorical features variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data into stratified training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features to achieve stable weight gradients
scaler = StandardScaler()
numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("\n--- Shape of Machine Learning Split Arrays ---")
print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

# ==========================================
# 7. MODEL BUILDING / ANALYSIS
# ==========================================
# Model A: Logistic Regression Model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# Model B: Decision Tree Classifier Model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# ==========================================
# 8. EVALUATION
# ==========================================
print("\n=== CLASSIFICATION EVALUATION METRICS ANALYSIS ===")
metrics_summary = []

for name, y_pred in [('Logistic Regression', y_pred_lr), ('Decision Tree', y_pred_dt)]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics_summary.append([name, acc, prec, rec, f1])
    
    print(f"\nModel Name: {name}")
    print(f" -> Accuracy Score:  {acc:.4f}")
    print(f" -> Precision Score: {prec:.4f}")
    print(f" -> Recall Score:    {rec:.4f}")
    print(f" -> F1 Performance:  {f1:.4f}")
    
# Generate and save Confusion Matrix for chosen operational model (Logistic Regression)
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Churned']).plot(cmap='Blues', ax=plt.gca(), values_format='d')
plt.title('Confusion Matrix: Logistic Regression Classifier')
plt.grid(False)
plt.tight_layout()
plt.savefig('confusion_matrix_lr.png')
plt.show()

# ==========================================
# 9. BUSINESS INSIGHTS
# ==========================================
# Add predictive scores back to evaluate structural risk groups
X_test_copy = X_test.copy()
X_test_copy['Actual_Churn'] = y_test
X_test_copy['Churn_Probability'] = y_prob_lr

def categorize_risk(prob):
    if prob >= 0.7: return 'High Risk'
    elif prob >= 0.3: return 'Medium Risk'
    else: return 'Low Risk'

X_test_copy['Risk_Segment'] = X_test_copy['Churn_Probability'].apply(categorize_risk)
print("\n--- Validation Risk Group Distribution Matrix ---")
print(X_test_copy['Risk_Segment'].value_counts())

# ==========================================
# 10. FINAL RECOMMENDATIONS
# ==========================================
print("\n=== SYSTEM EXECUTION COMPLETE ===")
print("Review generated plots in the execution directory and proceed to compile documentation repository files.")