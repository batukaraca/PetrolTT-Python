import pandas as pd  
import numpy as np  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

dataset = pd.read_csv("petrol_tuketimi.csv")  
dataset.head()  


X = dataset.iloc[:, 0:4].values  
y = dataset.iloc[:, 4].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

print('Ortalama Mutlak Hata:', metrics.mean_absolute_error(y_test, y_pred))  
print('Ortalama Karesel Hata:', metrics.mean_squared_error(y_test, y_pred))  
print('Ortalama Karekök Hatası:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  