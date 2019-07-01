# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:38:15 2019

@author: ZQQ

tools：Pycharm

用tensorflo框架搭建一个卷积神经网络
参考：

https://www.cnblogs.com/ansang/p/9164805.html

数据：http://download.tensorflow.org/example_images/flower_photos.tgz

"""

import tensorflow as tf

# 定义批量标准化函数,有效防止了梯度消失和爆炸，还加速了收敛
#def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
#     return tf.layers.batch_normalization(x,
#                                          momentum=momentum,
#                                          epsilon=epsilon,
#                                          scale=True,
#                                          training=train,
#                                          name=name)

def LeNet(x):
    ### 卷积，池化
    # 第一层卷积层（100->50）
    conv1 = tf.layers.conv2d(inputs=x,
                             filters=32,
                             #kernel_size=[3,3], # kernel_size = [5,5] 换不同的核大小，查看效果
                             kernel_size = [5,5],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))  # padding="same",卷积层不改变图片大小
    #conv1 = batch_norm(conv1, name="pw_bn1")  # 加入批量标准化
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) # 池化层图片大小缩小一半

    # 第二个卷积层(50->25)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv2 = batch_norm(conv2, name="pw_bn2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # 第三个卷积层(25->12)
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=128,
                             kernel_size=[3,3],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv3 = batch_norm(conv3, name="pw_bn3")
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2,2],
                                    strides=2)

    rel = tf.reshape(pool3,[-1, 12 * 12 * 128])

    # 防止过拟合，加入dropout
    #dropout = tf.layers.dropout(inputs=rel, rate=0.5)

    ### 全连接层
    dense1 = tf.layers.dense(inputs=rel,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    logits = tf.layers.dense(inputs=dense2,
                             units=5, # 5个类
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    pred = tf.nn.softmax(logits, name='prob') # softmax处理
    return logits, pred

### 四个卷积层，两个全连接层，一个softmax层组成。
### 在每一层的卷积后面加入batch_normalization, relu, 池化
### batch_normalization层很好用，加了它之后，有效防止了梯度消失和爆炸，还加速了收敛。
