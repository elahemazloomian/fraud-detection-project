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
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

fraud_data = pd.read_csv(data_path)
fraud = fraud_data.drop_duplicates()

x_fraud = fraud.drop(columns=["Class"])
y_fraud = fraud["Class"]


x_train, x_test, y_train, y_test = train_test_split(
    x_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)

reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)

print("----------------- Logistic Model ---------------------")
scaler_logestic = StandardScaler()
x_fraud_logestic_train = scaler_logestic.fit_transform(x_train)
x_fraud_logestic_test = scaler_logestic.transform(x_test)
logestic_model = LogisticRegression()
logestic_model.fit(x_fraud_logestic_train, y_train)

logestic_prediction_train = logestic_model.predict(x_fraud_logestic_train)
logestic_prediction_test = logestic_model.predict(x_fraud_logestic_test)
report_logestic = classification_report(y_test, logestic_prediction_test)
print(report_logestic)
cm_logestic = confusion_matrix(y_test, logestic_prediction_test)
cm_logestic_display = ConfusionMatrixDisplay(cm_logestic)
cm_logestic_display.plot()
plt.title("cm for logestic with 0.5 Threshold ")
plt.savefig(os.path.join(reports_dir, "logistic_cm.png"), bbox_inches='tight')
plt.show()

print("---------------- Threshold impact on Logistic -------------------------")
y_probs = logestic_model.predict_proba(x_fraud_logestic_test)[:, 1]
for th in [0.3, 0.5, 0.7]:
    y_pred_th = (y_probs >= th).astype(int)
    print(f"--- Threshold: {th} ---")
    print(classification_report(y_test, y_pred_th))

print("----------------- Unscaled KNN ---------------------")
k_neighbors = [1, 5, 20]
for i in range(0, 3):
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors[i])
    knn_model.fit(x_train, y_train)
    prediction_knn = knn_model.predict(x_test)
    print(f"k = {k_neighbors[i]}")
    print(classification_report(y_test, prediction_knn))
    cm_knn = confusion_matrix(y_test, prediction_knn)
    cm_knn_display = ConfusionMatrixDisplay(cm_knn)
    cm_knn_display.plot()
    plt.title(f"Unscaled KNN with k ={k_neighbors[i]}")
    plt.savefig(os.path.join(reports_dir, f"Unscaled_knn_cm_k{k_neighbors[i]}.png"), bbox_inches='tight')
    plt.show()

print("")
print("------------------- Scaled KNN ------------------")
knn_scaled = StandardScaler()
x_train_knn_scaled = knn_scaled.fit_transform(x_train)
x_test_knn_scaled = knn_scaled.transform(x_test)
k_neighbors = [1, 5, 20]
for i in range(0, 3):
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors[i])
    knn_model.fit(x_train_knn_scaled, y_train)
    prediction_knn_scaled = knn_model.predict(x_test_knn_scaled)
    print(f"KNN with k = {k_neighbors[i]}")
    print(classification_report(y_test, prediction_knn_scaled))
    cm_knn_scaled = confusion_matrix(y_test, prediction_knn_scaled)
    cm_knn_display_scaled = ConfusionMatrixDisplay(cm_knn_scaled)
    cm_knn_display_scaled.plot()
    plt.title(f"Scaled KNN with k ={k_neighbors[i]}")
    plt.savefig(os.path.join(reports_dir, f"Scaled_knn_cm_k{k_neighbors[i]}.png"), bbox_inches='tight')
    plt.show()

print("------------------- KNN Cross validation -------------------")
knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
evalutaion_metrics = ["accuracy", "recall", "precision", "f1"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_result = cross_validate(
    knn_pipeline,
    x_train,
    y_train,
    scoring=evalutaion_metrics,
    cv=cv,
)
for metric in evalutaion_metrics:
    mean_score = cv_result[f"test_{metric}"].mean()
    print(f"Mean {metric.capitalize()}: {mean_score:.4f} ")

print("----------------- Decision Tree Model -------------------------")
depth = [None, 2, 5, 10]
for i in range(4):
    print(f" max depth is {depth[i]}")
    desicion_model = DecisionTreeClassifier(max_depth=depth[i], random_state=42)
    desicion_model.fit(x_train, y_train)
    decision_prediction = desicion_model.predict(x_test)

    print(classification_report(y_test, decision_prediction))
    cm_decision_prediction = confusion_matrix(y_test, decision_prediction)
    cm_decision_display = ConfusionMatrixDisplay(confusion_matrix=cm_decision_prediction)
    cm_decision_display.plot()
    plt.title(f"Decision Tree Model with depth k ={depth[i]}")
    plt.savefig(os.path.join(reports_dir, f"tree_cm_depth_{depth[i]}.png"), bbox_inches='tight')
    plt.show()

print("-------------------- Tree Cross validation --------------------")
desicion_pipeline = make_pipeline(
    DecisionTreeClassifier(random_state=42)
)
evalutaion_metrics = ["accuracy", "recall", "precision", "f1"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_result = cross_validate(
    desicion_pipeline,
    x_train,
    y_train,
    scoring=evalutaion_metrics,
    cv=cv,
)
for metric in evalutaion_metrics:
    mean_score = cv_result[f"test_{metric}"].mean()
    print(f"Mean {metric.capitalize()}: {mean_score:.4f} ")





models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(scaler_logestic, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(logestic_model, os.path.join(models_dir, "model.pkl"))

print(f"\nModel and Scaler successfully saved to '{models_dir}'!")