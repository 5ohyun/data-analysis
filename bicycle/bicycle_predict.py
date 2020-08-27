import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
import datetime
from sklearn.model_selection import cross_val_score


# train.csv를 이용해 count를 예측하는 모델을 만든후 test.csv의 count를 예측
# validation 과정 없음 , RandomForestRegressor


# 데이터 로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 타임 데이터 변환
train['datetime']=pd.to_datetime(train['datetime'])
test['datetime']=pd.to_datetime(test['datetime'])


for i in [train, test] :
    i["year"] = i["datetime"].dt.year
    i["month"] = i["datetime"].dt.month
    i["day"] = i["datetime"].dt.day
    i["hour"] = i["datetime"].dt.hour
    i["minute"] = i["datetime"].dt.minute
    i["second"] = i["datetime"].dt.second
    i["dayofweek"] = i["datetime"].dt.dayofweek


# 카테고리 변수들 카테고리타입으로 변환 
categorical_feature_names = ["season","holiday","workingday","weather", "dayofweek","month","year","hour"]

for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")
    
# 필요한 변수들 선택 (temp와 atemp 변수는 연속형 변수)
feature_names = ["season", "weather", "temp", "atemp","year", "hour", "dayofweek", "holiday", "workingday"] 


X_train = train[feature_names]

X_test = test[feature_names]
y_train = train["count"]

from sklearn.ensemble import RandomForestRegressor #count 예측, 회귀식 필요

max_depth_list = []

model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
test["count"] = predictions
test=pd.DataFrame(test)
test=test[['datetime','count']]

print(pd.DataFrame(test))


# test.csv의 datetime 순서대로 [datetime, count]를 가지는 dataframe을
# return 하도록 합니다.

