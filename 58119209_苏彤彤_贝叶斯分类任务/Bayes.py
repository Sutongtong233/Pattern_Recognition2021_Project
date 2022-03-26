import numpy as np
import pandas as pd
import math
import csv

train=pd.read_csv(r"train_data.csv")
test=pd.read_csv(r"test_data.csv")
#划分数据集
train_1=train[train['0']==1]
train_2=train[train['0']==2]
train_3=train[train['0']==3]
#对于每一个类别估计高斯参数。每一个类别用2*13的矩阵，存储13个特征的均值和方差。共3个矩阵
Gauss_1=np.zeros((2,13))
Gauss_2=np.zeros((2,13))
Gauss_3=np.zeros((2,13))

#先验概率
train=np.asarray(train)
prior=[0,0,0]
num=[0,0,0]
count=0
for i in range(train.shape[0]):
    count+=1
    if int(train[i][0])==1:
        num[0]+=1
    if int(train[i][0])==2:
        num[1]+=1
    if int(train[i][0])==3:
        num[2]+=1
prior[0]=num[0]/count
prior[1]=num[1]/count
prior[2]=num[2]/count

#条件概率密度估计
def evaluate(train):
    train=np.asarray(train)
    gauss = np.zeros((2, 13))
    for i in range(1,14):
        column=train[:,i]
        gauss[0,i-1]=np.average(column)
        gauss[1,i-1]=np.var(column)
    return gauss


Gauss_1=evaluate(train_1)
Gauss_2=evaluate(train_2)
Gauss_3=evaluate(train_3)

#返回三元组，分别是三个类的联合概率 sample为13维数组
def predict(sample):
    sample=np.asarray(sample)
    prob=[0,0,0]
    prob[0] += math.log2(prior[0])
    prob[1] += math.log2(prior[1])
    prob[2] += math.log2(prior[2])
    for i in range(13):
        prob[0]-=math.log2(pow(2*math.pi*Gauss_1[1][i],0.5))+(pow(sample[i]-Gauss_1[0][i],2)/2/Gauss_1[0][i])
        prob[1]-=math.log2(pow(2*math.pi*Gauss_2[1][i],0.5))+(pow(sample[i]-Gauss_2[0][i],2)/2/Gauss_2[0][i])
        prob[2]-=math.log2(pow(2*math.pi*Gauss_3[1][i],0.5))+(pow(sample[i]-Gauss_3[0][i],2)/2/Gauss_3[0][i])
    return prob

#输出到test_prediction
result=pd.DataFrame()
result[0] = 0
result[1]=0
result[2]=0
result[3]=0

pred_y = []

for i in range(test.shape[0]):
    test = np.asarray(test)
    prob=predict(test[i,1:])
    max_prob = max(prob)

    result.at[i,0]=round(prob[0])
    result.at[i,1] = round(prob[1])
    result.at[i,2] = round(prob[2])
    if max_prob==prob[0]:
        pred_y.append(1)
        result.at[i, 3] = 1
    if max_prob==prob[1]:
        pred_y.append(2)
        result.at[i, 3] = 2
    if max_prob==prob[2]:
        pred_y.append(3)
        result.at[i, 3] = 3

test_y=test[:,0]

print(test_y)
print(pred_y)
count=0
print('error:',abs(np.sum(test_y-pred_y)))
print('ACC:',((test.shape[0]-abs(np.sum(test_y-pred_y)))/test.shape[0]))

result.to_csv(r"test_prediction.csv", sep=',')

