# -*- coding: UTF-8 -*-
from __future__ import division  #加入后python2可以支持小数
import numpy as np
import random



#数据预处理函数  filename：输入文件名称
def data_preprocess(filename):
    infile = open(filename, 'r')
    data_set = []
    label = []
    for line in infile:
        reader = line.rstrip().split(',')
        if reader[0].find("@") == -1:
            data_set.append(reader)

    for i in data_set:
        label.append(i[3])
        del i[3]

    for i in data_set:
        i[0] = (float(i[0]) + 1.87)/2.835  # 0.965+1.87=2.835
        i[1] = (float(i[1]) + 0.228)/4.608  # 4.38+0.228=4.608
        i[2] = (float(i[2]) + 1.92)/2.441  # 1.92+0.521=2.441
    return (data_set,label)


#筛选测试集
def splict_test_and_train(data_set,testsize):
    data_set=np.array(data_set,dtype=float)
    test_data=[]
    train_data=[]
    num_Sample = data_set.shape[0] #即取data表中有多少个数据 ，即矩阵的第二维长度
    totall_index = range(num_Sample) #train_index是从0到样本数到数数矩阵
    test_len = testsize * num_Sample
    test_index = random.sample(totall_index,int(test_len)) #在train_index中随机取出30%的数据标签
    for i in test_index:
        test_data.append(data_set[i])
    train_index=list(set(totall_index)^set(test_index)) #求所有index和测试index的差集，即为训练集index
    for i in train_index:
        train_data.append(data_set[i])
    train_data=np.array(train_data,dtype=float)
    return (train_data,test_data,train_index,test_index)


#knn分类函数
def knn(input, train_data, label, k):  #input是输入的一条数据，包含三个属性值，train_data是训练集,label是标注数据list，k是knn的k值
    diecount=0
    livecount=0
    train_data = np.array(train_data, dtype=float)
    num_Sample = train_data.shape[0] #即取data表中有多少个数据 ，即矩阵的第二维长度
    diff = np.tile(input, (num_Sample, 1)) - train_data #在列方向上重复input的数据num_Sample次，行1次，从而使得一个行元素都相同的矩阵-data矩阵
    sqDiff = diff**2
    sumDiff = sqDiff.sum(axis=1) #list中每一个【a,b,c】内部的元素相加后得到一个数值 a+b+c,为新的list中一个元素
    distance = sumDiff**0.5
    rank = np.argsort(distance) #argsort函数返回的是数组中元素值从小到大的索引值，对于一维数组不用加axis
    final_label=[]
    for i in range(k):    #取距离最小的k个元素的索引值
        final_label.append(label[rank[i]]) #根据索引值找到对应的标签
    for j in final_label:   #找到数量最多的标签作为返回值
        if j == '-1.0':
            diecount += 1
        else:
            livecount += 1
    if diecount > livecount:
        result = '-1.0'
    else:
        result = '1.0'
    diecount = 0 #每一次判断完后归0
    livecount = 0 #每一次判断完后归0
    return result

#计算错误率函数
def mis(k,test_index,test_data,train_data,label):
    miscount = 0
    for i in range(len(test_data)):
        result = knn(test_data[i],train_data,label,k)
        if result == label[test_index[i]]:
            continue
        else:
            miscount += 1
    s=len(test_data)
    misratio = miscount/s*100
    return misratio

filename='titanic.dat'
testsize=0.3
data_set,label = data_preprocess(filename) #preprocess函数返回处理后的总数据和标签
train_data,test_data,train_index,test_index = splict_test_and_train(data_set,testsize)

klist=[1,3,5,7,9]
for i in klist:
    mistratio=mis(i,test_index,test_data,train_data,label)
    print "while k=%d, the ratio of mistake is %f " % (i,mistratio)


