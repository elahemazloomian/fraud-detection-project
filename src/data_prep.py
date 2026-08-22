import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

fraud_data = pd.read_csv(data_path)
print(fraud_data)
print(fraud_data.info())
print(fraud_data.describe())

print("--------------------- Checking duplicated --------------------")
print("")
print("dataframe shape before deleting duplicatations : ", fraud_data.shape)
print("")
fraud = fraud_data.drop_duplicates()
print("")
print("dataframe shape after deleting duplicatations : ", fraud.shape)
print("")
print("")
print("---------------- Checking Null values ------------------")
print("Number of Null values is : ", fraud.isna().sum())
print("")
print("")
print("-------------------- Checking data quality ----------------------")
negative_time_count = (fraud['Time'] < 0).sum()
negative_amount_count = (fraud['Amount'] < 0).sum()
print(f"Number of negative Time values: {negative_time_count}")
print(f"Number of negative Amount values: {negative_amount_count}")
if negative_time_count == 0 and negative_amount_count == 0:
    print("Data Quality Passed: No negative values found in Time or Amount.")
else:
    print("Data Quality Alert: Negative values detected!")

print("")
print("")
print("---------------- Checking data Imbalance ---------------")
class_legitmate = (fraud["Class"] == 0).sum()
print(f"Class legitmate numbers : {class_legitmate}")
class_fraud = (fraud["Class"] == 1).sum()
print(f"fraud Class numbers : {class_fraud}")
x=[class_legitmate,class_fraud]
y=["legitimate_class","fraud_class"]

plt.bar(y,x)
plt.yscale('log')
plt.xlabel("classes")
plt.ylabel(" number of transactions(log)")
plt.show()



print("------------------- Display Feature Corrolation -----------------")

plt.figure(figsize=(12, 9))
sns.heatmap(fraud.corr(), cmap="coolwarm", center=0, linewidths=0.1)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()




print("")
print("-------------- prepare dataset and test split-------------------")
x_fraud = fraud.drop(columns=["Class"])
y_fraud = fraud["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)

print(f"Dataset Split Complete! Train shape: {x_train.shape}, Test shape: {x_test.shape}")


