import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from win32cryptcon import X509_BITS_WITHOUT_TRAILING_ZEROES
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import skew
from category_encoders import TargetEncoder
import time
start_time = time.time()
from sklearn.preprocessing import OneHotEncoder

# load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.drop(columns=['Id'],inplace=True)
test_ids = test_df['Id']
test_df.drop(columns=['Id'],inplace=True)

y=np.log1p(train_df['SalePrice'])
X=train_df.drop(columns=['SalePrice'])

#features
categorical_features = X.select_dtypes(include = ['object']).columns
numerical_features = X.select_dtypes(exclude = ['object']).columns

#handling missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numerical_features] = num_imputer.fit_transform(X[numerical_features])
X[categorical_features]=cat_imputer.fit_transform(X[categorical_features])

test_df[numerical_features] = num_imputer.transform(test_df[numerical_features])
test_df[categorical_features] = cat_imputer.transform(test_df[categorical_features])


#handling skewed data
skewed_features = X[numerical_features].apply(lambda x:skew(x.dropna()))
high_skew = skewed_features[skewed_features >0.75].index
X[high_skew] =np.log1p(X[high_skew])
test_df[high_skew] = np.log1p(test_df[high_skew])

#new features
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['HouseAge'] = X['YrSold'] - X['YearBuilt']
X['Remodeled'] = (X['YearBuilt'] != X['YearRemodAdd']).astype(int)
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['Remodeled'] = (test_df['YearBuilt'] != test_df['YearRemodAdd']).astype(int)

# Encode categorical variables using Target Encoding
encoder = TargetEncoder()
X[categorical_features] = encoder.fit_transform(X[categorical_features],y).astype('float32')
test_df[categorical_features] = encoder.transform(test_df[categorical_features]).astype('float32')

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.25,random_state=42)

#preprocessing
preprocessor = ColumnTransformer([
    ('num',StandardScaler(),numerical_features),],remainder='passthrough')
#base models
rf_model = RandomForestRegressor(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
gb_model = HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=6,random_state=42)
ridge_model= Ridge(alpha= 1,  solver='saga',max_iter=5000)

#stacking models
stacked_model = StackingRegressor(
    estimators=[
        ('rf',rf_model),
        ('gb',gb_model),
        ('ridge',ridge_model)
    ],
    final_estimator=XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=6)
)

stacked_model.fit(X_train,y_train)

valid_pred = stacked_model.predict(X_valid)
rmse= np.sqrt(mean_squared_error(y_valid,valid_pred))
print(f'Stacked Model Validation RMSE: {rmse}')

#cross validation score
cv_score = cross_val_score(stacked_model,X,y,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
rmse_scores = np.sqrt(-cv_score)
print(f'Cross-Validation RMSE: {rmse_scores.mean()}')

end_time = time.time()  # Record end time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

# Prepare test data predictions
test_preds = stacked_model.predict(test_df)
test_preds = np.expm1(test_preds)  # Reverse log transformation

# Create a submission file
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_valid, valid_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual SalePrice (log-transformed)')
plt.ylabel('Predicted SalePrice (log-transformed)')
plt.title('Predictions vs. Actual Sale Prices')
plt.legend()
plt.show()


#Stacked Model Validation RMSE: 0.15609814983386555
#Cross-Validation RMSE: 0.13996422684430873

# Stacked Model Validation RMSE: 0.16447631758499287
# Cross-Validation RMSE: 0.1457060605763964
# Execution Time: 175.0047 seconds