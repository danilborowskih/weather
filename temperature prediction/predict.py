import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel('weather.xls', skiprows=6)
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)
data = data[data['T'].notna()] # переписываем без Nane


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
# Scatter - график из точек, а не из линий
#plt.figure(figsize=(20, 5))
#plt.scatter(X_train['dayofyear'], y_train, label='Data train')
#plt.scatter(X_test['dayofyear'], y_test, label='Data test')
#plt.legend()
#plt.show()

# 1) LinearRegression + dayofyear
def Line():
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Посмотрим, какую мат. модель построила регрессия по данным, поэтому распечатаем прогноз для тренировочных данных
    pred_train = model.predict(X_train)
    # Прогноз на данных, которые модель еще не видела
    pred_test = model.predict(X_test)
    plt.figure(figsize=(20, 5))
    plt.scatter(X_train['dayofyear'], y_train, label='Data train')
    plt.scatter(X_test['dayofyear'], y_test, label='Data test')
    plt.scatter(X_train['dayofyear'], pred_train, label='Predict train')
    plt.scatter(X_test['dayofyear'], pred_test, label='Predict test')
    plt.legend()
    plt.show()
    # Проверяем качество численно
    # mean_squared_error - средняя сумма квадратов отклонений (меньше -> лучше)
    print('Суммарная ошибка на обучающей выборке =', mean_squared_error(y_train, pred_train))
    print('Суммарная ошибка на тестовой выборке =', mean_squared_error(y_test, pred_test))


# Подключаем пакет numpy для математических функций
import numpy as np
# Посмотрим, в каком диапазоне находятся данные признака dayofyear
data['dayofyear'].min(), data['dayofyear'].max()
# Перешли к интервалу от 0 до 2pi
scaled = ((data['dayofyear'] - 1) / 365 ) * 2 * np.pi
scaled.min(), scaled.max(), 2 * np.pi
# Вернемся на шаг 1 feature generation - создаем новые признаки:
# scaled_dayofyear - сжимает диапазон [1, 366] -> [0, 2pi] - переходим в радианы
# cos_dayofyear - косинус дня в году (= длина проекции номера дня в году на горизонтальную ось, если нпредставить номера циферблатом)
#data['cos_dayofyear'] = np.cos(((data['dayofyear'] - 1) / 366 ) * 2 * np.pi)
#plt.plot(data['cos_dayofyear'])
#plt.show()
