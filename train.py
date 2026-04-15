import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load Data
df = pd.read_excel("C:\HR_project\project_hr_dataset.xlsb")

# 2. Preprocessing & Feature Engineering
df = df.drop(['Unnamed: 0', 'Employee_ID', 'Full_Name','Location', 'Status', 'Work_Mode'], axis=1)

def categorize_role(title):
    title_clean = str(title).strip()
    top_executive = ['CTO', 'CFO']
    director_level = ['Sales Director', 'Operations Director', 'HR Director']
    managerial_level = ['HR Manager', 'Finance Manager', 'IT Manager', 'Supply Chain Manager',
                        'Business Development Manager', 'Innovation Manager', 'Brand Manager', 'Account Manager']
    senior_specialist = ['Software Engineer', 'DevOps Engineer', 'Research Scientist',
                         'Data Analyst', 'SEO Specialist', 'Talent Acquisition Specialist',
                         'Content Strategist', 'Financial Analyst', 'Product Developer']
    
    if title_clean in top_executive: return 'Top-Executive'
    elif title_clean in director_level: return 'Director'
    elif title_clean in managerial_level: return 'Manager'
    elif title_clean in senior_specialist: return 'Senior/Specialist'
    else: return 'Junior/Entry_Level'

df['Job_Level'] = df['Job_Title'].apply(categorize_role)

# 3. Define Preprocessor
ordered_levels = ['Junior/Entry_Level', 'Senior/Specialist', 'Manager','Director', 'Top-Executive']

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe_dept', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Department']),
        ('oe_joblevel', OrdinalEncoder(categories=[ordered_levels]), ['Job_Level'])
    ]
)

# 4. Split and Transform
X = df[['Department', 'Job_Title','Job_Level','Performance_Rating','Experience_Years','Hire_Date']]
y = np.log(df['Salary_INR'])  # Log-scaling target

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit on training data ONLY to avoid data leakage
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# 5. Train Model
model = DecisionTreeRegressor(max_depth=None, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate (The Regression Metrics Logic)
y_pred_log = model.predict(X_test)

# Convert back to real currency for MAE and Accuracy calculation
y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred_log)

# Metrics Calculation
r2 = r2_score(y_test, y_pred_log)
mae = mean_absolute_error(y_test_real, y_pred_real)
# Using 1 - MAPE as a proxy for "Accuracy" percentage
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real))
accuracy_pct = (1 - mape) * 100

print(f"R-Squared: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Accuracy: {accuracy_pct:.2f}%")

# 7. Save Everything
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# This dictionary matches the keys in your Streamlit app
metrics_dict = {
    'r2': r2,
    'mae': mae,
    'accuracy': accuracy_pct 
}
joblib.dump(metrics_dict, 'metrics.pkl')

print("All files saved successfully!")