# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:18:08 2019

@author: ZQQ
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:54:57 2019

@author: ZQQ
"""

import time
from read_img import read_img, shuffle_data
import tensorflow as tf
from models import lenet
from batch_get_data import minibatches

start_time = time.time()

# 数据：http://download.tensorflow.org/example_images/flower_photos.tgz
# 花总共有五类，分别放在5个文件夹下。
path = 'flower_photos/' # 设置图片路径

# 设置超参数，准备将所有的图片resize成100*100
w = 100 # 宽度
h = 100 # 高度
c = 3  # 图片通道数

# step1: 加载数据集
print('step1:load the datasets...')

data, label = read_img(path)  # 调用read_img()函数,读取图片数据和对应的标签
x_train, y_train, x_val, y_val = shuffle_data(data,label) # 调用shuffle_data()函数，打乱数据集，并划分数据集

# step2: 构建模型并开始训练、测试
print('step2: build the model and training...')


    
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

logits, pred = lenet.LeNet(x) # pred 是经过softmax处理过的
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
#train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  # 和上面的优化器不同
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_epoches = 10
batch_size = 64

train_losses = []
train_acces = []
val_losses = []
val_acces = []
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

with open("log_lenet.txt", "w") as log_f:        
    for epoch in range(num_epoches):
        train_loss, train_acc, n_batch = 0, 0 , 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
            
        train_losses.append(train_loss / n_batch)
        train_acces.append(train_acc / n_batch)
        print('Epoch: %d - train loss: %.4f - train acc: %.4f' % (epoch, (train_loss / n_batch), (train_acc / n_batch)))
        
        log_f.write('Epoch: %d - train loss: %.4f - train acc: %.4f' % (epoch, (train_loss / n_batch), (train_acc / n_batch)))
        log_f.write('\n')
        log_f.flush()

        # validation
        val_loss, val_acc, n_batch = 0, 0 , 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
            
        val_losses.append(val_loss / n_batch)
        val_acces.append(val_acc / n_batch)
        #print('Epoch: %d' % epoch, '- validation loss: %.4f' % val_loss, '- validation acc: %.4f' % val_acc) # 没有除n_batch,错误额警示自己
        print('Epoch: %d - val loss： %.4f - val acc: %.4f' % (epoch,(val_loss / n_batch), (val_acc / n_batch))) # 为了体现两种输出，上面那种居然忘了除n_batch,找了好长时间bug！！！
        
        log_f.write('Epoch: %d - val loss： %.4f - val acc: %.4f' % (epoch,(val_loss / n_batch), (val_acc / n_batch)))
        log_f.write('\n')
        log_f.flush()
        
#        if epoch % 5 == 0:  
#            saver.save(sess, "result/model_save/save_net.ckpt",epoch)
#            print('Trained Model Saved.')
    
    sess.close()
        
end_time = time.time()
time = end_time - start_time
print('run time:',time)