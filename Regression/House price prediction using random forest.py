import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import time
start_time = time.time()

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Separate features and target
X = train_df.drop(columns=['SalePrice', 'Id'])
y = np.log1p(train_df['SalePrice'])
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Preprocessing
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
])

# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_valid_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
print(f'Validation RMSE: {rmse}')

# Prepare test data and generate predictions
test_X = test_df.drop(columns=['Id'])
test_preds = model.predict(test_X)
test_preds = np.expm1(test_preds)  # Reverse log transformation

end_time = time.time()

# Print elapsed time
print(f"Training Time: {end_time - start_time:.2f} seconds")