import os
import argparse
import mindspore as ms
from mindspore import context
import numpy as np
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
def create_dataset(dataset,batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 定义数据集
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    dataset = dataset.map(operations=type_cast_op, input_columns="labels", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=resize_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=rescale_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=rescale_nml_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=hwc2chw_op, input_columns="data", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch操作
    buffer_size = 10000
    mnist_ds = dataset.shuffle(buffer_size=buffer_size)
    mnist_ds = dataset.batch(batch_size, drop_remainder=True)

    return dataset

import mindspore.nn as nn
from mindspore.common.initializer import Normal
#MindSpore默认使用NCHW的格式
class myNet5(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=3):
        super(myNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(32, 64, 5, pad_mode='valid')
        self.fc1 = nn.Dense(64 * 5 * 5, 512, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(512, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 实例化网络
net = myNet5()
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=250, keep_checkpoint_max=16)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore.train.callback import Callback
#记录每轮的loss和acc
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = cb_params.cur_step_num

        if cur_step % 100 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            #打印或绘图训练过程
            self.steps_eval["step"].append(cur_step+cur_epoch*10)
            self.steps_eval["acc"].append(acc["Accuracy"])
            self.steps_loss["step"].append(cur_step+cur_epoch*10)
            self.steps_loss["loss"].append(round(float(str(cb_params.net_outputs)),3))
steps_loss = {"step": [], "loss": []}
steps_eval = {"step": [], "acc": []}


# 加载训练数据集
#batch转化
#解压
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        #dict={data:10000*3072 ndarray；labels：10000 list}
    return dict

#数据增强函数
def reinforce(dataset,num_parallel_workers=1):
    # 定义参数
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width))
    # rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    # rescale_op = CV.Rescale(rescale, shift)
    # hwc2chw_op = CV.HWC2CHW()
    # type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    # dataset = dataset.map(operations=type_cast_op, input_columns="labels", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=resize_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    # dataset = dataset.map(operations=rescale_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    # dataset = dataset.map(operations=rescale_nml_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    # dataset = dataset.map(operations=hwc2chw_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    return dataset
#打开pickle文件，并数据增强
batchs=[]
for i in range(1,6):
    batchs.append(unpickle("cifar-10-batches-py\data_batch_%s"%i))

test=unpickle("cifar-10-batches-py/test_batch")
#合并5batch
train_X_ls=[]
train_y=[]
for i in range(5):
    batch = batchs[i]
    for j in range(len(batch[b'labels'])):
        train_X_ls.append(batch[b'data'][j])
        train_y.append(batch[b'labels'][j])
test_X_ls=[]
test_y=[]
for i in range(len(test[b'labels'])):
    test_X_ls.append(test[b'data'][i])
    test_y.append(test[b'labels'][i])
#归一化
train_X_ls=(np.array(train_X_ls)/255).astype(('float32'))
train_y=np.array(train_y).astype('int32')
test_X_ls=(np.array(test_X_ls)/255).astype(('float32'))
test_y=np.array(test_y).astype('int32')
train_X=[]
test_X=[]
#reshape(3,32,32),数据增强（随机翻转）和label打包，给generatedataset传参
for i in range(50000):
    # if_flip=random.randint(0,1)#有一定的概率将图片翻转
    # if(if_flip==0):
    #     train_X.append(train_X_ls[i].reshape(3,32,32))
    # if(if_flip==1):
    #     temp=train_X_ls[i].reshape(3,32,32)
    #     temp=np.fliplr(temp)
    #     train_X.append(temp)
   train_X.append(train_X_ls[i].reshape(3, 32, 32))
ds_train=list(zip(train_X,train_y))
for i in range(10000):
    test_X.append(test_X_ls[i].reshape(3,32,32))
ds_test=list(zip(test_X,test_y))

#generate dataset  包含打乱数据，数据增强，batch化操作
trainset=ds.GeneratorDataset(source=ds_train,column_names=['data','labels'])
# trainset=reinforce(trainset)
trainset=trainset.shuffle(buffer_size=50000).batch(50,drop_remainder=True)
testset=ds.GeneratorDataset(source=ds_test,column_names=['data','labels'])
# testset=reinforce(testset)
testset=testset.shuffle(buffer_size=10000).batch(50,drop_remainder=True)

#模型训练
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
step_loss_acc_info = StepLossAccInfo(model ,testset, steps_loss, steps_eval)
print('start training...')
model.train(20, trainset,callbacks=[ckpoint,LossMonitor(100),step_loss_acc_info], dataset_sink_mode=False)
accuracy=steps_eval['acc']
loss=steps_loss['loss']
x_lable=steps_eval['step']

# 任务1：画图
p_acc=plt.subplot(1,2,1)
p_loss=plt.subplot(1,2,2)
#chart of acc
plt.sca(p_acc)
plt.plot(x_lable,accuracy)
plt.xlabel('total step')
plt.ylabel('Accuracy')
plt.title('chart of model accauracy')
#chart of loss
plt.sca(p_loss)
plt.plot(x_lable,loss)
plt.xlabel('total step')
plt.ylabel('Loss')
plt.title('chart of model loss')

plt.show()
#任务2：展示测试集,随机32张照片
#predict传参Tensor
#test_X(3,32,32)归一化array
import random
arg=np.arange(len(test_y))
random.shuffle(arg)
arg=arg[:32]
images=[]
labels=[]
pred=[]
for i in arg:
    images.append(test_X[i])
    labels.append(test_y[i])
    prob=model.predict(ms.Tensor(test_X_ls[i].reshape(1,3,32,32)))#返回的是各类别的概率值，取最大
    pred.append(np.argmax(prob.asnumpy(), axis=1))
for i in range(len(labels)):
    plt.subplot(4, 8, i+1)
    color = 'blue' if pred[i] == labels[i] else 'red'
    plt.title("pre:{}".format(pred[i]), color=color)
    plt.imshow(np.squeeze(images[i].transpose()))#(3,32,32)格式！
    plt.axis("off")
plt.show()







