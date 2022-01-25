import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_excel('changet_data.xlsx')#, skiprows=6)
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)
data = data[data['aver_T'].notna()] # переписываем без Nane


# Создаем новый признак - день в году
data['dayofyear'] = data['date'].dt.dayofyear
data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Вначале для простототы мы будем делать прогноз только на одном факторе - номере дня в году data['dayofyear']
# Но модель ожидает, что ей на вход придет двумерная таблица - 
# поэтому создаем из колонки date['T'] полноценную таблицу pandas DataFrame
X_train = pd.DataFrame()
X_train['dayofyear'] = data_train['dayofyear']  # X
X_test = pd.DataFrame()
X_test['dayofyear'] = data_test['dayofyear']
# "y" оставляем столбцом, как есть
y_train = data_train['T']
y_test = data_test['T']

# 2) LinearRegression + cos dayofyear
data['dayofyear'] = data['date'].dt.dayofyear
data['cos_dayofyear'] = np.cos(((data['dayofyear'] - 1) / 366 ) * 2 * np.pi)
# Заново переразбиваем датасет на train-test, чтобы изменения применились
condition = data['date'] < '2020-01-01'
data_train = data[condition]
data_test = data[data['date'] >= '2020-01-01']
X_train = pd.DataFrame()
X_train['cos_dayofyear'] = data_train['cos_dayofyear']  # заменяем признак
X_test = pd.DataFrame()
X_test['cos_dayofyear'] = data_test['cos_dayofyear']
# "y" оставляем столбцом, как есть
y_train = data_train['T']
y_test = data_test['T']

model = LinearRegression()
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
