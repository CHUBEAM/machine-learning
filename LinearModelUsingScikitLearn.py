import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

oced_bli = pd.read_csv('./datasets/lifesat/oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('./datasets/lifesat/gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1',
                             na_values='n/a')

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)  # 생략된 함수 prepare_country_stats
X = np.c_[country_stats['GDP per capita']]  # country_stats의 GDP per capita열을 1*n 행렬로 만듦
y = np.c_[country_stats['Life satisfaction']]

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)