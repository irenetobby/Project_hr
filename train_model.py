import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load your data 
df = pd.read_excel('C:\HR_project\project_hr_dataset.xlsb') 

# 2. Preprocessing
# We use ColumnTransformer to match the logic in your app.py
categorical_features = ['Department']
ordinal_features = ['Job_Level']

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe_dept', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('oe_joblevel', OrdinalEncoder(), ordinal_features)
    ]
)

# 3. Prepare Features and Target
X = df[['Department', 'Job_Title','Job_Level','Performance_Rating','Experience_Years','Hire_Date']]
# Your app logic uses np.exp(), so we must train on log-scaled target
y = np.log(df['Salary']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
X_train_transformed = preprocessor.fit_transform(X_train)
model = DecisionTreeRegressor()
model.fit(X_train_transformed, y_train)

# 5. Calculate Metrics for the Sidebar
y_pred = model.predict(preprocessor.transform(X_test))
mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
r2 = r2_score(y_test, y_pred)
# Dummy accuracy calculation since this is a regression task
accuracy = (1 - (mae / np.exp(y_test).mean())) * 100

metrics = {
    'mae': mae,
    'r2': r2,
    'accuracy': accuracy
}

# 6. SAVE AS BINARY (Crucial Step)
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(metrics, 'metrics.pkl')

print("Files generated successfully: model.pkl, preprocessor.pkl, metrics.pkl")