# train.py (outline)
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.scaler import StandardScalerFromScratch
from src.model import MyLinearRegression
from src.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal']).values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from-scratch pipeline
scaler = StandardScalerFromScratch()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = MyLinearRegression(alpha=0.01, n_iters=5000, verbose=True, random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
print("From-scratch MSE:", mean_squared_error(y_test, y_pred))
print("From-scratch R2:", r2_score(y_test, y_pred))

# sklearn pipeline
sc = StandardScaler().fit(X_train)
X_train_sk = sc.transform(X_train)
X_test_sk = sc.transform(X_test)
lr = LinearRegression().fit(X_train_sk, y_train)
y_pred_sk = lr.predict(X_test_sk)
print("sklearn MSE:", mean_squared_error(y_test, y_pred_sk))
print("sklearn R2:", r2_score(y_test, y_pred_sk))
