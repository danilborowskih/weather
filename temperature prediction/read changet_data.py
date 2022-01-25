#read changet_data
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_excel('changet_data.xlsx')
data_T=data['aver_T']
plt.plot(data['day'], data['aver_T'])
plt.show()