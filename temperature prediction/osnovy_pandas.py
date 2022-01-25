import pandas as pd
import matplotlib.pyplot as plt
#colab
# Считываем Excel-таблицу в переменную data и Удаляем лишние комментарии (первые 5 строк)
data = pd.read_excel('weather.xls', skiprows=6)
# Смотрим, что получилось
# первые 10 строк
#print(data.head(10))

# data.shape - атрибут
#print(data.shape) #показывает количество строк и колонн
# Посмотрим на колонки, которые нам доступны
#print(data.columns)

# Данные хранятся а) по столбцам б) по номерам строк
# Столбец - отдельный объект типа pandas.Series
#print(data['Местное время в Москве (ВДНХ)'])
#print(data['Местное время в Москве (ВДНХ)'][6])



# Хотим нарисовать график температуры от времени
# Сейчас там записаны строки в Российском формате - а нужно преобразовать во внутренний питоновский формат
"""
#pd.to_datetime('01.01.2022 03:00') --- Timestamp('2022-01-01 03:00:00') преобразует строку в формат временни
day = pd.to_datetime('06.01.2022 03:00') # В американском формате это читается как 1 июня, поэтому выдался 152 день в году
#print(day.dayofyear)#152
day = pd.to_datetime('06.01.2022 03:00', dayfirst=True)  # Теперь правильно
#print(day.dayofyear)# 6
"""
# Создали новую колонку в правильном формате
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)
#print(data.head(5))

def T_graph():
    """ Распечатаем график температуры"""
    x = data['date']
    y = data['T']
    plt.figure(figsize=(10, 5))
    plt.plot(x, y,label='f(t)=T',color='blue')
    plt.legend()
    plt.show()

# Нам доступно 100% данных или есть пропуски? Удаляем некорректные данные
#print(data[data['T'].isna()])# столбцы где есть Nane
data = data[data['T'].notna()] # переписываем без Nane

# Максимум, минимум, среднее
#print(data['T'].max(), data['T'].min(), data['T'].mean())

# Гистограмма - график, который показывает, сколько раз встречалось то или иное значение
data['T'].hist().plot()
#plt.show()

# "95-й квантиль равен 23.4 градусам" - если температура не превыает 23.4 градусов в 95% случаев
#print(data['T'].quantile(0.95), data['T'].quantile(0.05)) # В 90% случаев температура находится в диапазоне от <-9.9> до <23.4> градусов


# Выбираем данные по условию
condition = (data['date'] < '2018-03-18') & (data['date'] > '2018-03-08')
data_short = data[condition]
#data_short = data[data['date'].between('2018-03-18','2018-03-08')

x = data_short['date']
y = data_short['T']
plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.show()