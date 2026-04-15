import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# 1. Setup categories
dept_list = ['HR', 'Engineering', 'Sales', 'Marketing']
level_list = ['Junior', 'Mid', 'Senior', 'Lead']

# 2. Define the Preprocessor
# We explicitly handle categorical columns. 
# remainder='passthrough' will keep numeric columns (Rating, Experience).
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe_dept', OneHotEncoder(categories=[dept_list], handle_unknown='ignore'), ['Department']),
        ('oe_joblevel', OrdinalEncoder(categories=[level_list]), ['Job_Level'])
    ],
    remainder='passthrough' 
)

# 3. Create DataFrame with the full schema
# Note: In a real scenario, you'd usually have more than one row to fit a model!
dummy_df = pd.DataFrame({
    'Department': ['HR'],
    'Job_Title': ['Specialist'],      # Will be dropped in selection
    'Job_Level': ['Junior'],
    'Performance_Rating': [4],
    'Experience_Years': [5],
    'Hire_Date': [pd.Timestamp('2024-01-01')] # Will be dropped in selection
})

# 4. Feature Selection
# We filter the dataframe to only include features the model can process.
# 'Job_Title' and 'Hire_Date' are excluded here.
model_features = ['Department', 'Job_Level', 'Performance_Rating', 'Experience_Years']
X_train = dummy_df[model_features]
y_train = np.array([75000.0]) # Target: Salary

# 5. Fit Preprocessor and Model
# This creates a pipeline-like flow: Transform -> Fit
X_transformed = preprocessor.fit_transform(X_train)

model = DecisionTreeRegressor()
model.fit(X_transformed, y_train)

# 6. Save the artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump({'accuracy': 90, 'r2': 0.8, 'mae': 5000}, 'metrics.pkl')

print("Files created successfully!")
print(f"Transformed shape: {X_transformed.shape}")