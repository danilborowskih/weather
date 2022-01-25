import pandas as pd
import datetime as dt
from Lagrange import Lagrange
data = pd.read_excel('weather.xls', skiprows=6,)
#data = pd.read_excel('weather.xls', skiprows=6,index_col='Местное время в Москве (ВДНХ)')
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)
#data = data[data['T'].notna()] # переписываем без NaN
#data = data.drop([4315])

data['hour'] = data['date'].dt.hour #создаем столб часов
data['day'] = data['date'].dt.date # столб даты без времени

aver_T_list = [] # размер будет равен количеству строк в таблице
dara_days = [str(data['day'][4314]) if i == 4315 else str(data['day'][i]) for i in range(len(data['day']))]
data['T'][4315]=data['T'][4314]
print(dara_days)
for day in dara_days:
    day_T = []
    day_hour = []
    for i in range(len(aver_T_list), len(dara_days)) :
        if day == dara_days[i]:
            day_T.append(data['T'][i])
            day_hour.append(data['hour'][i])
        else: break
    
    perhaps_list = [Lagrange(day_hour, day_T, x) for x in range(24)]
    aver_T_list.extend([round(sum(perhaps_list)/24, 2)]*len(day_T))
#print(len(aver_T_list), aver_T_list)
data['aver_T'] = aver_T_list
data.to_excel("changet_data.xlsx")