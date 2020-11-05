# 随机森林
import numpy  as np
import pandas as pd

# 随机森林
from sklearn.ensemble import RandomForestClassifier

# 获取数据
datas = pd.read_csv("complete_data_set.csv")
datas = datas.drop('Unnamed: 0',1).values
X = datas[:,:-1]
Y = datas[:,-1]


clf = RandomForestClassifier(criterion='gini',max_depth=8,max_features='log2',n_estimators=120)

# 保存模型
from sklearn.externals import joblib

# 保存模型
joblib.dump(clf,'RandomForest.pickle')
