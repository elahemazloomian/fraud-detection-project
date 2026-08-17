# Credit Card Fraud Detection Pipeline

A machine learning pipeline designed to detect fraudulent credit card transactions using scaled feature representations, cross-validation, and model persistence.

---

## Project Structure

```text
first mini project/
│
├── data/
│   └── creditcard.csv          # Raw transaction dataset
├── models/
│   ├── model.pkl               # Saved trained Logistic Regression model
│   └── scaler.pkl              # Saved StandardScaler object
├── reports/
│   ├── logistic_cm.png         # Logistic regression confusion matrix
│   ├── knn_cm_k5.png           # KNN evaluation plot
│   └── tree_cm_depth_5.png     # Decision Tree evaluation plot
├── src/
│   ├── data_prep.py            # Data loading & cleaning pipeline
│   ├── train.py                # Model training, CV, & artifact export
│   └── predict.py              # Inference script for new samples
├── .gitignore
├── requirements.txt
└── README.md



Models Evaluated

Logistic Regression: Scaled features with probability threshold tuning (0.3, 0.5, 0.7).

K-Nearest Neighbors (KNN): Evaluated both unscaled and scaled inputs across k = [1, 5, 20].

Decision Tree Classifier: Tested varying max depths [None, 2, 5, 10].

Stratified K-Fold Cross Validation: Evaluated pipeline performance using 5-fold CV on Accuracy, Recall, Precision, and F1-score.