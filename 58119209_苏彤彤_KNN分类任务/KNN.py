import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import csv

train=pd.read_csv(r'train_data.csv')
test=pd.read_csv(r'val_data.csv')
testtest=pd.read_csv(r'test_data.csv')

#马氏距离训练结果
A0=np.array([[-0.00550996,-0.00154484,  0.00351539 , 0.0097689 ],
 [ 0.00482415, -0.00265759, -0.00327927 ,-0.00371036]])

def Euclid(a,b):
    dis=0
    for i in range(len(a)):
        dis+=pow(a[i]-b[i],2)
    return dis
def Chebyshev(a,b):
    dis=[]
    for i in range(len(a)):
        dis.append(abs(a[i]-b[i]))
    return max(dis)
def Manhattan(a,b):
    dis=0
    for i in range(len(a)):
        dis+=abs(a[i]-b[i])
    return dis
def Mahalanobis(A,a,b):
    M=np.dot(A.T,A)
    dis=np.dot((a-b),M)
    dis = np.dot(dis,(a-b).T)
    return dis


#创建结果存储dataframe,在最后加一列
testtest["My_prediction"]=0
print(type(testtest))

def predict(train,test,method,k):
    #转化为ndarray进行处理
    train=np.asarray(train)
    test=np.asarray(test)
    # 数据划分
    train_X0 = train[:, :4]
    train_y = train[:, 4]
    test_X0 = test[:, :4]
    test_y = test[:, 4]
    # Z-score标准化(正态)
    # train_X = (train_X0 - train_X0.mean()) / (train_X0.std())
    # test_X = (test_X0 - test_X0.mean()) / (test_X0.std())

    #minmax标准化
    train_X0=train_X0.T
    test_X0=test_X0.T
    train_X=np.zeros(shape=(train_X0.shape[0],train_X0.shape[1]))
    test_X = np.zeros(shape=(test_X0.shape[0], test_X0.shape[1]))

    for i in range(train_X0.shape[0]):
        max_train=max(train_X0[i,:])
        min_train=min(train_X0[i,:])
        train_X[i,:]=(train_X0[i,:]-min_train)/(max_train-min_train)
        test_X[i,:]=(test_X0[i,:]-min_train)/(max_train-min_train)
    train_X = train_X.T
    test_X = test_X.T

    result=[]#预测结果
    test_count=0
    for i in test_X:
        test_count+=1
        dis = []  # 每个测试样本和所有训练样本的距离
        for j in train_X:
            if method=='Euclid':
                dis.append((Euclid(i, j)))
            if method=='Chebyshev':
                dis.append((Chebyshev(i, j)))
            if method == 'Manhattan':
                dis.append((Manhattan(i, j)))
            if method=='Mahalanobis':
                dis.append((Mahalanobis(A0,i,j)))
        sort_dis = np.argsort(dis)
        sort_dis = sort_dis[:k]
        count_1=count_0=0
        for i in sort_dis:
            if train_y[i]==1:count_1+=1
            else:count_0+=1
        if count_0>count_1:
            result.append(0)
            # testtest.at[test_count,"My_prediction"]=0
        else:
            result.append(1)
            # testtest.at[test_count, "My_prediction"] = 1
    #结果存入csv
    # if method=='Euclid':
    #     testtest.to_csv("task1_test_Euclidean.csv",sep=",")
    # if method == 'Chebyshev':
    #     testtest.to_csv("task1_test_Chebyshev.csv", sep=",")
    # if method == 'Manhattan':
    #     testtest.to_csv("task1_test_Manhattan.csv", sep=",")
    # if method == 'Mahalanobis':
    #     testtest.to_csv("task2_test_ prediction.csv", sep=",")
    rightnum = 0
    test_y = test[:, 4]
    for i in range(test_count):
        if result[i] == test_y[i]: rightnum += 1
    accuracy = rightnum / test_count
    return accuracy
#对测试集合
# predict(train,testtest,"Euclid",3)
# predict(train,testtest,"Chebyshev",3)
# predict(train,testtest,"Manhattan",3)
# predict(train,testtest,"Mahalanobis",3)

def evaluate(test,method):
    # 准确率
    test = np.asarray(test)
    test_count=test.shape[0]
    rightnum = 0
    result=predict(train,test,method,3)
    test_y = test[:, 4]
    for i in range(test_count):
        if result[i] == test_y[i]: rightnum += 1
    accuracy = rightnum / test_count
    return accuracy

#打印结果
# print("基于欧氏距离验证集准确率：",evaluate(test,"Euclid"))
# print("基于切比雪夫距离验证集准确率：",evaluate(test,"Chebyshev"))
# print("基于曼哈顿距离验证集准确率：",evaluate(test,"Manhattan"))
# print("基于马氏距离验证集准确率：",evaluate(test,"Mahalanobis"))
ls_1=[]
ls_2=[]
ls_3=[]
ls_4=[]
#验证集测试
#多种K的取值
k_list=[1,3,5,15,35]
for i in k_list:
    ls_1.append(predict(train,test,"Euclid",i))
    ls_2.append(predict(train,test,"Chebyshev",i))
    ls_3.append(predict(train,test,"Manhattan",i))
    ls_4.append(predict(train,test,"Mahalanobis",i))

#画图
data=np.zeros(shape=(len(k_list),4))
ls_1=np.asarray(ls_1).transpose()
ls_2=np.asarray(ls_2).transpose()
ls_3=np.asarray(ls_3).transpose()
ls_4=np.asarray(ls_4).transpose()
data[:,0]=ls_1
data[:,1]=ls_2
data[:,2]=ls_3
data[:,3]=ls_4
data1=pd.DataFrame(data=data,index=k_list,columns=['Euclid','Chebyshev',"Manhattan","Mahalanobis"])
sns.set_palette(sns.hls_palette(5,0.44,0.69))
sns.lineplot(data=data1,markers=True)
plt.show()



#task 2
e=2

scaler = lambda x :  (x - x.mean()) / (x.std())
train[['Recency (months)']]=train[['Recency (months)']].apply(scaler)
train[['Frequency (times)']]=train[['Frequency (times)']].apply(scaler)
train[['Monetary (c.c. blood)']]=train[['Monetary (c.c. blood)']].apply(scaler)
train[['Time (months)']]=train[['Time (months)']].apply(scaler)
t1=train[train['whether he/she donated blood in March 2007']==1]
t0=train[train['whether he/she donated blood in March 2007']==0]
num1=len(t1)
num0=len(t0)
t1=np.asarray(t1)
t0=np.asarray(t0)
train=np.asarray(train)
t1=t1[:,:4]
t0=t0[:,:4]

#学习马氏距离
def val_grad(A):
    result=np.zeros((2,4))
    for i in train:
        if(i[4]==1):
            X_i=i[:4]
            p=np.zeros((2,4))
            q=0
            for k in t1:
                p-=math.exp(-pow(np.linalg.norm(A*k-A*X_i),2))*A*(k-X_i)*(k-X_i).T
                q+=math.exp(-pow(np.linalg.norm(A*k-A*X_i),2))
            result-=(num1-1)*(p)/(q)
            for j in t1:
                result-=A * (j - X_i) * (j - X_i).T
        if(i[4]==0):
            X_i = i[:4]
            p = np.zeros((2, 4))
            q = 0
            for k in t0:
                p -= math.exp(-pow(np.linalg.norm(A*k-A*X_i), 2)) * A * (k - X_i) * (k - X_i).T
                q += math.exp(-pow(np.linalg.norm(A*k-A*X_i), 2))
            result -= (num0 - 1) * (p) / (q)
            for j in t0:
                result -= A * (j - X_i) * (j - X_i).T
        else:
            pass
    return result
def GD(A,lr,t):
    count=0
    for i in range(t):
        count+=1
        A+=lr*val_grad(A)
        # print(count)
        print(A)
    print(A)

# A=0.1*np.random.standard_normal(size = (2,4))
# GD(A,0.00001,100)
# A0=np.array([[-0.00550996,-0.00154484,  0.00351539 , 0.0097689 ],
#  [ 0.00482415, -0.00265759, -0.00327927 ,-0.00371036]])

# predict(train,test,'Euclid',1)

#
# p = np.zeros((2, 4))
# print(p)
# i=np.array([1,2,3,4]).T
# sample=np.array([2,3,4,5]).T
# A=np.array([[1,2,3,4],[2,3,4,5]])
#
# print(A*(i-sample)*(i-sample).T)
