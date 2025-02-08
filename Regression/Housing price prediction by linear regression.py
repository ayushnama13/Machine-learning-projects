import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from debugpy.adapter.components import missing
from jsonschema.exceptions import best_match
from send2trash.util import preprocess_paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# print(train_df.info())
# print(train_df.describe())

missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending = False)


X = train_df.drop(columns = ["SalePrice","Id"])
y = train_df["SalePrice"]

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

num_features = X_train.select_dtypes(include=['int64','float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scalar',StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num',num_pipeline,num_features)
    ,('cat',cat_pipeline,cat_features)
])

models = {
    'Linear Regression':LinearRegression(),
    'Ridge Regression':Ridge(alpha = 1),
    'Lasso Regression' : Lasso(alpha = 0.01)
}
best_model = None
best_rmse = float('inf')
model_performance = {}
for name,model in models.items():
    pipeline = Pipeline([
        ('preprocessor',preprocessor)
        ,('model',model)
    ])
    pipeline.fit(X_train,y_train)
    y_pred=pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}\n")
