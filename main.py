import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Paths
train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Check for dataset
if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("❌ Please download 'train.csv' and 'test.csv' from Kaggle and place them in the 'data/' folder.")
    exit()

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Target and drop unused columns
y = np.log1p(train['SalePrice'])
X = train.drop(['Id', 'SalePrice'], axis=1)
test_ids = test['Id']
test = test.drop(['Id'], axis=1)

# Imputation
def handle_missing(df):
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    for col in df.select_dtypes(include='number'):
        df[col] = num_imputer.fit_transform(df[[col]]).ravel()

    for col in df.select_dtypes(include='object'):
        df[col] = cat_imputer.fit_transform(df[[col]]).ravel()

    return df

X = handle_missing(X)
test = handle_missing(test)

# Feature engineering
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['Age'] = 2025 - X['YearBuilt']
X['QualAge'] = X['OverallQual'] * X['Age']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
test['Age'] = 2025 - test['YearBuilt']
test['QualAge'] = test['OverallQual'] * test['Age']

# One-hot encode
X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

# Align train and test
X, test = X.align(test, join='left', axis=1, fill_value=0)

# Save full feature list
features = X.columns.tolist()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Evaluate
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"✅ Validation MSE: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# Save model, scaler, features
os.makedirs("model", exist_ok=True)
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Artifacts saved in /model")
