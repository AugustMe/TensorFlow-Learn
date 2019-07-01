# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:23:46 2019

@author: ZQQ
"""

import numpy as np
import os
import glob
from skimage import transform,io
import random

# 定义超参数
w = 100 
h = 100

#### 定义读取图片的函数：read_img()
def read_img(path):
    data_list = [path + x for x in os.listdir(path) if os.path.isdir(path + x)] # 所有图片分类目录
    imgs = [] # 定义一个imgs空列表，存放遍历读取的图片
    labels = [] # 定义一个labels空列表，存放图片标签
    for idx, folder in enumerate(data_list):  # 遍历每个文件夹中的图片，idx表示
        for im in glob.glob(folder + '/*.jpg'):  # *:匹配0个或多个字符
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h)) # 将所有图片的尺寸统一为:100*100(宽度*高度)
            with open('datasets_name.txt','w') as f:
                f.write(folder+im+'_'+str(idx)+'\n')
            imgs.append(img) # 遍历后更改尺寸后的图片添加到imgs列表中
            labels.append(idx) # 遍历后更改尺寸后的图片标签添加到labels列表中
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32) # np.float32是类型 后面两个变量是没有进行np.asarray

# np.asarray 和 np.array
# np.array与np.asarray功能是一样的，都是将输入转为矩阵格式。
# 主要区别在于 np.array（默认情况下）将会copy该对象，而 np.asarray除非必要，否则不会copy该对象。

### 定义随机打乱数据集的函数：shuffle_data()
def shuffle_data(data,label):
    # 打乱顺序
    data_size = data.shape[0] # 数据集个数
    arr = np.arange(data_size) # 生成0到datasize个数
    np.random.shuffle(arr) # 随机打乱arr数组
    data = data[arr] # 将data以arr索引重新组合
    label = label[arr] # 将label以arr索引重新组合
    
#    # 打乱数据顺序的另一种方法，当然还有其他的方法
#    index = [i for i in range(len(data))]
#    random.shuffle(index)
#    data = data[index]
#    label = label[index]
    
    # 将所有数据分为训练集和验证集
    ratio = 0.8 # 训练集比例
    num = np.int(len(data) * ratio)
    x_train = data[:num]
    y_train = label[:num]
    x_val = data[num:]
    y_val = label[num:]
    
    return x_train, y_train, x_val, y_val

#path = 'flower_photos/' # 所有图片的总路径(目录)
#data, label = read_img(path)
