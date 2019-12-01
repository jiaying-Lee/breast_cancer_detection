# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import random

np.set_printoptions(suppress=True)  # 结果有e时可以直接输入该函数使得e可以被计算出来


def data_preprocess(filename):
    # 打开文件
    infile = open(filename, 'r')
    # 初始化储存数组
    data_set = []
    label = []

    # 逐行阅读文件，将非标注数据存入data_set
    for line in infile:
        reader = line.rstrip().split(',')
        if reader[0].find("@") == -1:
            data_set.append(reader)

    # 将data_set数据分别存入data_set、label数组中，data_set只有属性值
    for i in data_set:
        label.append(i[3])
        del i[3]
    data_set = np.array(data_set, dtype=float)
    label = np.array(label, dtype=float)

    # 将标签从-1，1转为0，1表示
    for i in range(len(label)):
        if label[i] == -1:
            label[i] = 0
        else:
            continue

    return (data_set, label)


# sigmoid 单元处理函数，输入：输入数据data，是否求导数（默认为false）
def sigmoid(x, deriv=False):
    if (deriv == True):  # 如果需要对sigmoid求导数，则其导数为x*(1-x)
        return x * (1 - x)
    else:  # 如果不需要对sigmoid求导数，则其输出sigmoid函数1/(1+np.exp(-x))
        return 1 / (1 + np.exp(-x))


# 分割测试集和训练集
def splict_test_and_train(data_set, label, testsize):

    data_set = np.array(data_set, dtype=float)
    test_data = []
    train_data = []
    test_label = []
    train_label = []

    num_Sample = data_set.shape[0]  # 即取data表中有多少个数据 ，即矩阵的第二维长度
    totall_index = range(num_Sample)  # train_index是从0到样本数到数数矩阵
    test_len = testsize * num_Sample
    test_index = random.sample(totall_index, int(test_len))  # 在train_index中随机取出30%的数据标签

    for i in test_index:
        test_data.append(data_set[i])
        test_label.append(label[i])
    train_index = list(set(totall_index) ^ set(test_index))  # 求所有index和测试index的差集，即为训练集index

    # 根据取的训练集index取出样本数据存入train_data
    for i in train_index:
        train_data.append(data_set[i])
        train_label.append(label[i])

    train_data = np.array(train_data, dtype=float)
    test_data = np.array(test_data, dtype=float)
    train_label = np.array(train_label, dtype=float)
    test_label = np.array(test_label, dtype=float)
    train_label = train_label.reshape(len(train_label), -1)  # 将train_label倒置
    test_label = test_label.reshape(len(test_label), -1)  # 将test_label倒置

    return (train_data, test_data, train_label, test_label)


# 批量梯度下降法
def BGD_train(layer1size, layer2size, layer3size, iteration, train_data, train_label, studyratio):
    np.random.seed(1)

    # 初始化各层单元之间的权值，即输入层到隐藏层，隐藏层到输出层，分别是w1，w2
    # np.random.random()  默认生成 0~1 之间的小数，若要指定生成[a,b)的随机数，(b-a)*np.random()+a
    w1 = -1 + 2 * np.random.random((layer1size, layer2size))  #生成 layer1size*layer2size 的 -1~1 之间的随机数矩阵
    w2 = -1 + 2 * np.random.random((layer2size, layer3size))  #生成 layer2size*layer3size 的 -1~1 之间的随机数矩阵

    for j in xrange(iteration):

        #正向传播
        output0 = train_data    #output0为n*3维矩阵，xij表示第i个样本的第j个属性取值
        datapass01=np.dot(output0, w1) #输入层的数据经过w1加权后的矩阵。datapass01的xij代表第i个样本的属性分别依次乘上对应路径的权重后求和结果，为隐藏层第j个单元的输入值。这里我们的输入层因为有三个属性，对应三个单元，隐藏层有4个单元，所以该矩阵为n*4维
        output1 = sigmoid(datapass01)   #对上述datapass01进行sigmoid函数处理。矩阵中每一个元素都为上一个矩阵的同一位置元素处理后的值，所以该矩阵为n*4维
        datapass02=np.dot(output1, w2) #隐藏层的数据经过w2加权后的矩阵。datapass02的xij代表第i个样本在隐藏层单元上的输出分别依次乘上对应路径的权重后求和结果，为输出层第j个单元的输入值，这里我们的输出层只有一个单元，所以该矩阵为n*1维
        output2 = sigmoid(datapass02)   #对上述datapass02进行sigmoid函数处理。矩阵中每一个元素都为上一个矩阵的同一位置元素处理后的值，所以该矩阵为n*1维

        #反向传播
        diff = train_label - output2    #求模型输出的标签值（即output2）与实际标签的差值，diff为n*1维矩阵，xi1表示第j个样本预测值与真实值的差
        error2 = diff * sigmoid(output2, deriv=True) #由公式：输出层单元k误差项=ok(1-ok)(tk-ok) 得。diff为n*1维，output2在sigmoid导数处理后仍然为n*1维，我们想要求每一个diff(i)和output2.sigmoid(i)的乘积，用*表示两个矩阵对应位置的数乘积，得出error2仍为n*1维度
        error1sum = np.dot(error2,w2.T)#为下面求和准备的矩阵error1sum：（每一个输出单元k误差项*k到h反向传播路径的权值），error2为n*1维，w2为4*1维，所以需要将w2转置，error1sum为n*4维，其中xij=第i个样本的输出误差*从隐藏层第j个单元到输出层单元的路径权值
        error1 = error1sum * sigmoid(output1, deriv=True) #error1sum为n*4维，output1在sigmoid导数处理后仍然为n*4维，两个矩阵对应元素相乘，error1为n*4维，xij=第i个样本的输出误差*隐藏层第j个单元到输出单元路径权值*这个j单元输出值的sigmoid导数处理

        #更新权值
        w2 += np.dot(output1.T,error2) * studyratio# studyratio一般取0.01-0.1。由公式delta（wji）=学习率*误差项*xji得。output1为n*4维，转置后为4*n维，error2为n*1维，则权值更新矩阵为4*1维，和w2相同，直接相加即可更新权值。其中更新矩阵的xi1=求和（每个样本从隐藏层第i个单元输出值*每个样本对应的error2对应误差项）*学习率
        w1 += np.dot(output0.T, error1) * studyratio #由公式：隐藏层单元h误差项=oh(1-oh)*求和（每一个输出单元k误差项*k到h反向传播路径的权值）得。ouptut0为3*n维，转置后为n*3维，error1为n*4维，更新矩阵为3*4维，和w1相同，直接相加即可更新权值，其中更新矩阵的xij=求和（每一个样本的第i个属性（即输入值）*处理这个属性的对应隐藏层第j个单元到输出层单元的路径权值*该样本的error2误差项（即对应的输出单元的误差项）*该样本在该单元输出值的sigmoid导出处理值）

        # 每1000次打印出所有样本的预测平均误差
        if (j % 1000) == 0:
            diff=np.abs(diff)
            mean=np.mean(diff)
            print "The BGD mean error on test data: %f " % mean

    return w1, w2

#随机梯度下降法
def SGD_train(layer1size, layer2size, layer3size, iteration, train_data, train_label, studyratio):

    np.random.seed(1)
    # 初始化各层单元之间的权值，即输入层到隐藏层，隐藏层到输出层，分别是w1，w2
    w1 = -1 + 2 * np.random.random((layer1size, layer2size))
    w2 = -1 + 2 * np.random.random((layer2size, layer3size))

    # 根据用户填入迭代次数进行迭代
    for i in range(iteration):

        #打乱训练集，模拟随机
        totall_index = range(len(train_data))  # totall_index是从0到训练集样本数的index顺序矩阵
        random_index = random.sample(totall_index, len(train_data))  # 将totall_index随机打乱来依次训练

        #每一次迭代，都会根据所有打乱顺序对训练集样本依次训练，每取一个样本后更新一次权值，此循环算法与BGD基本一样，只有输入数据量之差
        for j in random_index:

            #正向传播
            output0 = train_data[j]
            datapass01 = np.dot(output0, w1)
            output1 = sigmoid(datapass01)
            datapass02 = np.dot(output1, w2)
            output2 = sigmoid(datapass02)

            #反向传播
            diff = train_label[j] - output2
            error2 = diff * sigmoid(output2, deriv=True)
            error1sum = np.dot(error2, w2.T)
            error1 = error1sum * sigmoid(output1, deriv=True)

            #由于输入的只有一条样本数据，导致一下的输出为list，无法进行矩阵运算，这里把它们都化为矩阵形式
            output1 = np.matrix(output1)
            error2 = np.matrix(error2)
            output0 = np.matrix(output0)
            error1 = np.matrix(error1)

            #更新权值
            w2 += np.dot(output1.T, error2) * studyratio  # 一般取0.01-0.1
            w1 += np.dot(output0.T, error1) * studyratio

        # 每10次打印出所有样本的预测平均误差
        if (j % 2) == 0:
            diff=np.abs(diff)
            mean=np.mean(diff)
            print "The SGD mean error on test data: %f " % mean

    return w1, w2

#测试模型
def bp_test(test_data, w1, w2):

    #输入测试数据进入模型进行向前传播
    output0 = test_data
    output1 = sigmoid(np.dot(output0, w1))
    output2 = sigmoid(np.dot(output1, w2))

    #初始化result，用于存储预测的结果类标
    result = []
    #如果预测结果大于0.5则预测类标为1，小于0.5则预测类标为0
    for i in range(len(output2)):
        if output2[i] > 0.5:
            result.append(1)
        else:
            result.append(0)
    #返回对测试集的所有样本预测结果
    return result


# 计算错误率函数
def mis(result, test_label):
    #初始化miscount用于记录错误预测次数
    miscount = 0
    #判断每一个测试样本是否预测错误
    for i in range(len(result)):
        if result[i] == test_label[i]:
            continue
        else:
            miscount += 1
    #计算错误率
    s = len(test_label)
    misratio = miscount / s * 100
    return misratio

#读入数据
filename = 'titanic.dat'
data_set, label = data_preprocess(filename)
#划分测试集和训练集
train_data, test_data, train_label, test_label = splict_test_and_train(data_set, label, 0.3)
#将训练集数据分别代入BGD，SGD进行训练
BGD1, BGD2 = BGD_train(3, 4, 1, 5000, train_data, train_label, 0.03)
SGD1, SGD2 = SGD_train(3, 4, 1, 10, train_data, train_label, 0.025)
#得出结果进行准确率计算
BGDresult = bp_test(test_data, BGD1, BGD2)
BGDaccuracy = 100 - mis(BGDresult, test_label)
SGDresult = bp_test(test_data, SGD1, SGD2)
SGDaccuracy = 100 - mis(SGDresult, test_label)
#打印结果
print "The BGD accuracy on test data: %f " % BGDaccuracy
print "The SGD accuracy on test data: %f " % SGDaccuracy
