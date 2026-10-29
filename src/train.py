import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]

    # Split into Reference (train) and Current (simulated production) data
    ref_data, curr_data = train_test_split(df, test_size=0.3, random_state=42)

    # Train model
    X_train = ref_data.drop(columns=["target"])
    y_train = ref_data["target"]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save artifacts
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    ref_data.to_csv("data/reference.csv", index=False)
    curr_data.to_csv("data/current.csv", index=False)
    joblib.dump(model, "models/model.joblib")
    
    print("Training complete. Artifacts saved.")

if __name__ == "__main__":
    main()