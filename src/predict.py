import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")


scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

def predict_sample(sample_df):
    """Scales features and predicts fraud status."""
    sample_scaled = scaler.transform(sample_df)
    predictions = model.predict(sample_scaled)
    probabilities = model.predict_proba(sample_scaled)[:, 1]
    return predictions, probabilities

if __name__ == "__main__":
    
    data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    sample_data = pd.read_csv(data_path, nrows=5)
    X_sample = sample_data.drop(columns=["Class"])
    y_true = sample_data["Class"]

    predictions, probabilities = predict_sample(X_sample)

    print("\n--- INFERENCE RESULTS ---")
    for idx, (pred, prob, true_val) in enumerate(zip(predictions, probabilities, y_true)):
        status = "Fraud" if pred == 1 else "Legitimate"
        actual = "Fraud" if true_val == 1 else "Legitimate"
        print(f"Sample {idx+1}: Predicted = {status} (Prob: {prob:.4f}) | Actual = {actual}")