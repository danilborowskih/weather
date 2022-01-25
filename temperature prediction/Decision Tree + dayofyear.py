import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
data = pd.read_excel('weather.xls',skiprows=6)
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)
data = data[data['T'].notna()] # переписываем без Nane

# Создаем новый признак - день в году
data['dayofyear'] = data['date'].dt.dayofyear
data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']


# 3) Decision Tree + dayofyear

data['dayofyear'] = data['date'].dt.dayofyear
# data['cos_dayofyear'] = np.cos(((data['dayofyear'] - 1) / 366 ) * 2 * np.pi)

# Заново переразбиваем датасет на train-test, чтобы изменения применились
data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']

X_train = pd.DataFrame()
X_train['dayofyear'] = data_train['dayofyear']  # заменяем признак
X_test = pd.DataFrame()
X_test['dayofyear'] = data_test['dayofyear']
# "y" оставляем столбцом, как есть
y_train = data_train['T']
y_test = data_test['T']

model = DecisionTreeRegressor()  # Заменяем модель
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

plt.figure(figsize=(20, 5))
plt.scatter(data_train['dayofyear'], y_train, label='Data train')
plt.scatter(data_test['dayofyear'], y_test, label='Data test')
plt.scatter(data_train['dayofyear'], pred_train, label='Predict train')
plt.scatter(data_test['dayofyear'], pred_test, label='Predict test')
plt.legend()

print('Суммарная ошибка на обучающей выборке =', mean_squared_error(y_train, pred_train))
print('Суммарная ошибка на тестовой выборке =', mean_squared_error(y_test, pred_test))

# 1) LinearRegression + dayofyear
# Суммарная ошибка на обучающей выборке = 101.10651961303319
# Суммарная ошибка на тестовой выборке = 108.68372355444073
# 2) LinearRegression + cos dayofyear
# Суммарная ошибка на обучающей выборке = 31.334653858869338
# Суммарная ошибка на тестовой выборке = 32.252821295486534
# 3) Decision Tree + dayofyear
# Суммарная ошибка на обучающей выборке = 20.78417741544515
# Суммарная ошибка на тестовой выборке = 34.74021538620518