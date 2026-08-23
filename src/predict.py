import os
import sys
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

THRESHOLD = 0.5
EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def predict_sample(sample_df, threshold=THRESHOLD):
    """Scales features and predicts fraud status for one or more rows."""
    sample_scaled = scaler.transform(sample_df)
    probabilities = model.predict_proba(sample_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def predict_from_json(input_data: dict, threshold=THRESHOLD) -> dict:
    """
    Takes one transaction as a dict, returns a result matching the
    project's required output.json schema.
    """
    missing = [f for f in EXPECTED_FEATURES if f not in input_data]
    if missing:
        return {
            "prediction": None,
            "class_id": None,
            "probability": None,
            "threshold": threshold,
            "status": f"error: missing fields {missing}",
        }

    try:
        df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
        predictions, probabilities = predict_sample(df, threshold)
        class_id = int(predictions[0])
        probability = float(probabilities[0])

        return {
            "prediction": "Fraud" if class_id == 1 else "Legitimate",
            "class_id": class_id,
            "probability": round(probability, 4),
            "threshold": threshold,
            "status": "success",
        }
    except Exception as e:
        return {
            "prediction": None,
            "class_id": None,
            "probability": None,
            "threshold": threshold,
            "status": f"error: {e}",
        }


if __name__ == "__main__":
   
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    with open(input_path) as f:
        input_data = json.load(f)

    result = predict_from_json(input_data)

    print(json.dumps(result, indent=2))

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
