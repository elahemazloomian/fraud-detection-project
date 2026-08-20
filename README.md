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
│   ├── Scaled_knn_cm_k5.png    # KNN evaluation plot
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


Engineering Analysis & Key Findings

Catastrophic Impact of Unscaled Features:


When features are not scaled, high-magnitude variables (Amount and Time) dominate the Euclidean distance calculations. Consequently, as K increases from 1 to 20 without scaling, Recall drops drastically from 14.74% down to 0.00%. At K=20, the model fails to detect a single fraudulent transaction because minority class instances are completely overwhelmed by nearest neighbors from the majority class.

Performance Recovery via Feature Scaling:
Applying StandardScaler standardizes all features to zero mean and unit variance. This leads to a massive improvement across all metrics:

F1-Score Boost: F1-score increases from 0.0412 to 0.7975 at K=5.

Recall Improvements: Recall jumps from 2.11% to 68.42% for K=5.

Hyperparameter (K) Selection:

K=1: Yields the highest Recall (71.58%) with scaling, but suffers from lower Precision (82.93%) due to sensitivity to individual noisy data points.

K=5: Provides the optimal balance, achieving the highest F1-Score (0.7975) and exceptional Precision (95.59%).