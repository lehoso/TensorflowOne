#%%

#读取MNIST数据
## 在conda python环境中，tensorflow 1.15可能没有examples文件
##在\Lib\site-packages\tensorflow\文件夹下没有examples文件
##就在tensroflow_core文件中复制examples
from tensorflow.examples.tutorials.mnist import input_data

mnist_data_folder = "../MNIST_data/"
data = input_data.read_data_sets(mnist_data_folder, one_hot=False)

#%%

#划分数据集
##准备训练数据，验证数据，测试数据
X0 = data.train.images
Y0 = data.train.labels
X1 = data.validation.images
Y1 = data.validation.labels
X2 = data.test.images
Y2 = data.test.labels

print(X0.shape)

#%%

# 手写数字展示
from matplotlib import pyplot as plt

plt.figure()
fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
for i in range(10):
    Im = X0[Y0 == i][0].reshape(28, 28)
    ax[i].imshow(Im)
plt.show()

#%%

#产生one_hot型因变量
print(Y0)
from keras.utils import to_categorical

YY0 = to_categorical(Y0)
YY1 = to_categorical(Y1)
YY2 = to_categorical(Y2)
print(YY0)

#%%

#模型构建
from keras.layers import Input, Dense, Activation
from keras import Model

input_shape = (784,)
input_layer = Input(input_shape)
x = Dense(10)(input_layer)
x = Activation("softmax")(x)
output_layer = x

model = Model(input_layer, output_layer)
model.summary()

#%%

#模型编译
from keras.optimizers import Adam

model.compile(optimizer=Adam(0.01),
              loss="categorical_crossentropy",
              metrics=["accuracy"])  #默认值是0.1

#%%

#模型拟合
history = model.fit(X0, YY0,
                    validation_data=(X1, YY1),
                    batch_size=1000,
                    epochs=10)

#%%

#调用model.fit返回一个history对象。这个对象有个成员history，
# 它是一个字典，里面包含训练过程中所有的数据
history_dict = history.history
history_dict.keys()

#%%

#查看实验损失函数图
import matplotlib.pyplot as plt

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)
#绘制训练损失函数
plt.plot(epochs, loss_values, color="blue", label="train_loss")
plt.plot(epochs, val_loss_values, color="red", label="validation_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()  #给图像加图例
plt.show()

#%%

#import matplotlib.pyplot as plt

loss_acc = history_dict["accuracy"]
val_loss_acc = history_dict["val_accuracy"]

epochs = range(1, len(loss_acc) + 1)
#绘制精确度变化曲线图
plt.plot(epochs, loss_acc, "b", color="red", label="train_loss")
plt.plot(epochs, val_loss_acc, "b", color="blue", label="validation_loss")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend()  #给图像加图例
plt.show()

#%%

#以上的模型针对于10分类问题精度已经达到93%

result = model.evaluate(X2, YY2)  #输入数据和标签，返回损失和精确度
result

#%%

#查看第二层的权重参数
a = model.layers[1].get_weights()
a
model.layers[1].get_weights()[0].shape


#%%

#预测一下测试集里面的第二张图片
import numpy as np

x = X2[1]
x = x.reshape(1, 784)
y_pre_pic2 = model.predict(x)
y_pre_pic2 = np.argmax(y_pre_pic2)
y_pre_pic2

#%%

Y2[1]

#%%

#计算预测值与真实值的精确度
from sklearn import metrics

y_pre = model.predict(X2)
#y_pre.shape
y_pre_labels = []
for i in range(len(y_pre)):
    y_labels = np.argmax(y_pre[i])
    y_pre_labels.append(y_labels)

# mse=((y_pre_label-Y2)**2).mean()
len(y_pre_labels)

acc = metrics.accuracy_score(Y2, y_pre_labels)
acc

#%%
#参数可视化
from matplotlib import pyplot as plt

plt.figure()
fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
weights = model.layers[1].get_weights()[0]
for i in range(10):
    Im = weights[:, i].reshape((28, 28))
    ax[i].imshow(Im, cmap='seismic')
    ax[i].set_title("{}".format(i))
    ax[i].set_xticks({})
    ax[i].set_yticks({})
plt.show()

