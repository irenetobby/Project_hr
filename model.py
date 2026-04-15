import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor  
def train_model():
    # Load dataset
    df = pd.read_excel("C:\HR_project\project_hr_dataset.xlsb")

    # Target column
    target_column = "Salary_INR"

    # Features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train model - Using Decision Tree Regressor
    # You can add random_state=42 to ensure the results are the same every time you run it
    model = DecisionTreeRegressor(random_state=42) 
    model.fit(X, y)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save column names
    with open("columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print("✅ Decision Tree Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()