# 5 - Classifier example
# import os
# os.environ['KERAS_BACKEND']='theano' #底层运用theano搭建神经网络

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # normalize将像素值标准化为0-1之间
X_test = X_test.reshape(X_test.shape[0], -1) / 255  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)  # label为10，分为10中类型
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build two-layer neural net
model = Sequential([
    Dense(32, input_dim=784),  # output_dim为32个features，input_dim为784即每一个像素点作为一个神经元
    Activation('relu'),

    Dense(10),  # 输出10个feature
    Activation('softmax'),
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # learning rate为0.001

# We add metrics to get more results you want to see
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',  # 损失函数
    metrics=['accuracy'])  # 用一个矩阵保存准确率，还有一些其他的参数******

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)