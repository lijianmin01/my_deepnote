import torch.nn as nn
import torch
import torch.utils.data as Data

import numpy as np
import pandas as pd

# 获取数据
def get_data():
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    # X = min_max.fit_transform(X)
    datas = pd.read_csv("complete_data_set.csv")
    datas = datas.drop('Unnamed: 0', 1).values
    # 随机打乱数据
    np.random.shuffle(datas)
    # 取前200个样本作为测试数据，其余的作为训练数据
    data =datas[:,:-1]
    labels = datas[:,-1]
    labels = np.array([int(i) for i in labels])
    train = datas[200:-4,:-1]
    train = min_max.fit_transform(train)
    test = datas[:200,:-1]
    test = min_max.fit_transform(test)
    train_labels = datas[200:-4,-1].reshape(-1,1)
    test_labels = datas[:200,-1].reshape(-1,1)
    return train,train_labels,test,test_labels,data,labels

# 构建神经网络模型
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        # 添加神经元以及函数
        self.fc=nn.Sequential(
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 10),
            # nn.ReLU(),
            # nn.Linear(72, 9)
        )
        # 定义损失函数
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(),lr=0.1)

    # 向前传播
    def forward(self,inputs):
        outputs =self.fc(inputs)
        return outputs

    # 训练数据
    def train(self, x ,labels):
        # 正向传播
        out = self.forward(x)
        # 计算损失
        loss = self.mse(out,labels)
        # 梯度清零
        self.optim.zero_grad()
        # 计算新的梯度
        loss.backward()
        # 根据新的梯度更新参数
        self.optim.step()


    def test(self,test_):
        return self.fc(test_)


if __name__ == '__main__':
    train, train_labels, test, test_labels, data, labels = get_data()
    mynet = mynet()
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(),torch.from_numpy(train_labels).long())
    BATCH_SIZE = 10
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(50):
        for step, (x, y) in enumerate(train_loader):
            y = torch.reshape(y, [BATCH_SIZE])
            mynet.train(x, y)
            # if epoch % 20 == 0:
            #     print('Epoch: ', epoch, '| Step: ', step, '| batch y: ', y.numpy())
            # if epoch % 1000 == 0:
            #     print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))
    out = mynet.test(torch.from_numpy(data).float())
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    test_y = labels.reshape(1, -1)
    target_y = torch.from_numpy(test_y).long().data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

    print(pred_y)
    print(target_y[0])
    print("预测准确率", accuracy)







