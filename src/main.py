import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import aurum as au

logging.basicConfig(level=logging.DEBUG)

au.use_datasets("black_friday.csv")

bf_file = "black_friday.csv"

bf = pd.read_csv(bf_file)

au.parameters(test_size=0.15, random_state=0, n_estimators=1000)

X = bf.iloc[:, 0:6].values
y = bf.iloc[:, 9].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=au.test_size, random_state=au.random_state)

#################################
# Encoding non-numerical columns
x_train_encoder = LabelEncoder()
X_train[:, 0] = x_train_encoder.fit_transform(X_train[:, 0])
X_train[:, 1] = x_train_encoder.fit_transform(X_train[:, 1])
X_train[:, 3] = x_train_encoder.fit_transform(X_train[:, 3])
X_train[:, 4] = x_train_encoder.fit_transform(X_train[:, 4])

x_test_encoder = LabelEncoder()
X_test[:, 0] = x_test_encoder.fit_transform(X_test[:, 0])
X_test[:, 1] = x_test_encoder.fit_transform(X_test[:, 1])
X_test[:, 3] = x_test_encoder.fit_transform(X_test[:, 3])
X_test[:, 4] = x_test_encoder.fit_transform(X_test[:, 4])

######################
# Scaling all columns
X_train_scaler = StandardScaler()
X_test_scaler = StandardScaler()

X_train = X_train_scaler.fit_transform(X_train)
X_test = X_test_scaler.fit_transform(X_test)

#################################
# Training and error measurement
regressor = RandomForestRegressor(n_estimators=au.n_estimators, random_state=au.random_state)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
error = mean_absolute_error(y_test, y_pred)

au.register_metrics(error=error)
au.end_experiment()
