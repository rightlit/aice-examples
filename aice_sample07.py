import pandas as pd
df = pd.read_csv('HousingData.csv')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

model.score(X_test, y_test)
model2.score(X_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred)

y_pred = model2.predict(X_test)
mean_squared_error(y_test, y_pred)

import numpy as np
accuracy_score(y_test.values, y_pred)
