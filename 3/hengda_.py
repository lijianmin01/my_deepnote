import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

column_names = ['性别','婚姻状况','年龄','教育程度',
                '职业','工作年限','双收入','家庭成员',
                '18岁以下','户主状况','家庭类型','民族类别',"双收入"
                '语言','收入等级']

data = pd.read_csv('train.csv',names=column_names)
data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
data = data.astype('int')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data[column_names],data[column_names[13]],test_size=0.25,random_state=33)

from sklearn import neural_network
clf = neural_network.MLPClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

cnt = 0
y_test = y_test.values.tolist()
y_pred = list(y_pred)
for i in range(len(y_test)):
    if(y_test[i]==y_pred[i]):
        cnt+=1

print("ac"+str(cnt*1.0/len(y_test)))
