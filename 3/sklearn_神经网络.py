# 利用cross_val_score自动获得结果
import numpy  as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 获取数据
from sklearn.neural_network import MLPClassifier
column_names = ['性别','婚姻状况','年龄','教育程度',
                '职业','工作年限','双收入','家庭成员',
                '18岁以下','户主状况','家庭类型','民族类别',"双收入"
                '语言','收入等级']
datas = pd.read_csv("train.csv",names=column_names).values

X = datas[:, :-1]
Y = datas[:, -1]
# 分层划分数据，采用五折家产验证
# sfolder = StratifiedKFold(n_splits=100, shuffle=True, random_state=13)

# 激活函数
activation_list = ['logistic', 'tanh', 'relu']
# 求解器
solver_list = ['lbfgs', 'sgd', 'adam']
alpha_list = [0.0001,0.01,1,100,1000]

max_ = 0.0
string = None

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=33)

clf = MLPClassifier()
# 采用五折交叉验证
clf.fit(X_train,y_train)
test_prv=clf.predict(X_test)
avg_s = accuracy_score(test_prv,y_test)
print("AC->{}".format(avg_s))

# for activation in activation_list:
#     for solver in solver_list:
#         for alpha in alpha_list:
#             clf = MLPClassifier(random_state=13,max_iter=300,activation=activation,solver=solver,alpha=alpha)
#             # 采用五折交叉验证
#             clf.fit(X_train,y_train)
#             test_prv=clf.predict(X_test)
#             avg_s = accuracy_score(test_prv,y_test)
#             print("{}  {}  {}   ->{}".format(activation, solver, alpha, avg_s))
#             if max_ < avg_s:
#                 max_ = avg_s
#                 string = ("{}  {}  {}   ->{}".format(activation, solver, alpha, avg_s))
#
# print("最好： " + string)

