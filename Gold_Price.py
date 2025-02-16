import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
gold_data=pd.read_csv("/content/gld_price_data.csv")
gold_data.head()
gold_data.shape
gold_data.tail()
gold_data.info()
gold_data.describe()
correlation=gold_data.corr()
X=gold_data.drop(['Date',"GLD"],axis=1)
Y=gold_data["GLD"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
regressor=RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
test_data_prediction=regressor.predict(X_test)
error_score=metrics.r2_score(Y_test,test_data_prediction)
print(error_score)
training_data_prediction=regressor.predict(X_train)
error_score=metrics.r2_score(Y_train,training_data_prediction)
print(error_score)
