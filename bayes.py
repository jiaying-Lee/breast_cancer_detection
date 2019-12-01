# -*- coding: UTF-8 -*-
from __future__ import division  # 加入后python2可以支持小数
import numpy as np
import random

#数据预处理
def data_preprocess(filename):
    infile = open(filename, 'r')    #打开文件
    #初始化储存数组
    data_set = []
    label = []
    feature1 = []
    feature2 = []
    feature3 = []

    #逐行阅读文件，将非标注数据存入data_set
    for line in infile:
        reader = line.rstrip().split(',')
        if reader[0].find("@") == -1:
            data_set.append(reader)

    #将data_set数据分别存入对应属性、标签数组中
    for i in data_set:
        label.append(i[3])
        feature1.append(i[0])
        feature2.append(i[1])
        feature3.append(i[2])
        del i[3]
    #将三个属性list和总data_set转换为数组，label为list，方便后面处理
    data_set = np.array(data_set, dtype=float)
    feature1 = np.array(feature1, dtype=float)
    feature2 = np.array(feature2, dtype=float)
    feature3 = np.array(feature3, dtype=float)
    label = np.array(label, dtype=float)
    label = list(label)
    #返回三个属性数组、总数据数组、标签list
    return (data_set, label, feature1, feature2, feature3)

#根据信息增益找到属性的分点
def find_split_piont(feature):
    #找到分点的区域
    norepeat = list(set(feature))  # 得到第一个属性中含有的值，即把重复的值全都只保留一个
    norepeat = np.array(norepeat, dtype=float)
    norepeat.sort() #将所包含的值从小到大排序

    point_entropy = []
    #根据feature的取值个数，循环（取值个数-1）次，每一次以第i个值和第i+1个值的中点作为分点进行计算
    for i in range(len(norepeat) - 1):
        Leftcount = 0
        Ldiecount = 0
        Llivecount = 0

        Rightcount = 0
        Rlivecount = 0
        Rdiecount = 0

        # 该循环计算对于一个分点，在分点左边点的个数，右边点的个数，以及两边分别的生存和死亡数
        for j in range(len(feature)):
            if feature[j] <= norepeat[i]:
                Leftcount += 1
                if label[j] == -1.0:
                    Ldiecount += 1
                else:
                    Llivecount += 1
            else:
                Rightcount += 1
                if label[j] == -1.0:
                    Rdiecount += 1
                else:
                    Rlivecount += 1

        totalcount = Leftcount + Rightcount #总个数=左边总个数+右边总个数
        Leftfre = Leftcount / totalcount #在分点左边的频率=在左边的个数/总个数
        Rightfre = Rightcount / totalcount #在分点右边的频率=在右边的个数/总个数
        Llivefre = Llivecount / Leftcount #在分点左边的点标签为存活的频率=左边存活的个数/左边的总个数
        Ldiefre = Ldiecount / Leftcount  #在分点左边的点标签为死亡的频率=左边死亡的个数/左边的总个数
        Rlivefre = Rlivecount / Rightcount #在分点右边的点标签为存活的频率=右边存活的个数/右边的总个数
        Rdiefre = Rdiecount / Rightcount #在分点右边的点标签为死亡的频率=右边死亡的个数/右边的总个数
        Lentropy = ((-1) * Llivefre * (np.log2(Llivefre)) + ((-1) * Ldiefre * (np.log2(Ldiefre)))) #根据熵计算公式计算左边的熵
        Rentropy = ((-1) * Rlivefre * (np.log2(Rlivefre)) + ((-1) * Rdiefre * (np.log2(Rdiefre)))) #根据熵计算公式计算右边的熵
        entropy = Leftfre * Lentropy + Rightfre * Rentropy #总熵
        point_entropy.append(entropy) #把每一次计算的总熵存到point_entrpoy中

    rank = np.argsort(point_entropy) #按照point_entropy中的值从小到大排序，找到最小的熵值对应的索引号
    split_point = (norepeat[rank[0] + 1] + norepeat[rank[0]]) / 2 #计算出分裂点的值 即中点（信息熵最小值点+后一个点）/2
    print split_point
    return (point_entropy, split_point)

def split_feature(split_point,feature):
    #初始化list用于存储分裂后的属性值
    split_feature=[]
    #对每一个属性值进行判，在阈值左边的是0，右边的是1，存入split_feature，实现离散化
    for i in feature:
        if i < split_point:
            j = 0
            split_feature.append(j)
        else:
            j = 1
            split_feature.append(j)
    return(split_feature)


def splict_test_and_train(data_set,testsize):

    #将原数据list转array
    data_set=np.array(data_set,dtype=float)

    #初始化，test_data=[]用于存储测试数据，train_data=[]用于存储训练数据
    test_data=[]
    train_data=[]

    #按testsize随机取的训练集和测试集
    num_Sample = data_set.shape[0] #即取data表中有多少个数据 ，即矩阵的第二维长度
    totall_index = range(num_Sample) #train_index是从0到样本数到数数矩阵，根据输入数据的数量，随机生成0到数据量-1的所有数字的随机排列
    test_len = testsize * num_Sample
    test_index = random.sample(totall_index,int(test_len)) #在train_index中随机取出30%的数据标签

    #根据取的测试集index取出样本数据存入test_data
    for i in test_index:
        test_data.append(data_set[i])
    train_index=list(set(totall_index)^set(test_index)) #求所有index和测试index的差集，即为训练集index

    # 根据取的训练集index取出样本数据存入train_data
    for i in train_index:
        train_data.append(data_set[i])
    train_data=np.array(train_data,dtype=float)
    #返回训练数据、测试数据、训练索引、测试索引
    return (train_data,test_data,train_index,test_index)



def bayes(label,input_feature,train_data,train_index):
    #初始化p_y 字典，用于存储整体数据中标签分别的概率，输出为p（y=1）=0.3230349840981372  p(y=-1)=0.6769650159018628
    p_y = {}

    #计算所有标签分别的在整个数据集的占比
    for i in label:
        p_y[float(i)] = label.count(i)/len(label)

    # 初始化字典P_xy,以feature(x)=a|y=b的方式存放输入数据中，在确定类别的条件下，每一个属性取值概率，一共有3*2=6个p值
    P_xy = {}

    #根据每一个可能分类结果分别进行迭代计算可能性，分y=0，y=1两次迭代
    for y in p_y.keys():
        # 从label中找到取值为0（第二次为1）的分别的样本的索引
        y_index =[i for i, search_label in enumerate(label) if search_label == y]
        # 对于每一个属性进行迭代
        for j in range(len(input_feature)):
            # 对于第j个属性，从train_data中找到取值和所给对象的j个属性一样的样本的索引号
            x_index1 = [i for i, search_feature in enumerate(train_data[:,j]) if search_feature == input_feature[j]]
            #初始化x_index，储存取值与输入样本属性值一样的样本在data_set的索引
            x_index=[]
            # 根据取值同对象一样的样本的train_data索引号去找到该样本在data_set中的索引号，使得这里y_index和x_index相等时，对应的样本是同一个
            for k in range(len(x_index1)):
                x_index.append(train_index[x_index1[k]])
            #xy_count用于计算与输入样本属性值和标签一样的样本个数
            xy_count = len(set(x_index) & set(y_index))  # x_index，y_index求并集，列出两个表相同的元素
            #pkey是P_xy的key值，为feature(x)=属性值|y=标签
            pkey = 'feature'+ str(j+1)+'='+str(input_feature[j]) + '|y=' + str(y)
            #P_xy用于存储，在属于类别b的情况下，属性值x=a的后验概率p(feature(x)=a|y=b)
            P_xy[pkey] = xy_count / float(len(y_index)) #p(feature(x)=a|y=b)=（feature(x)=a|y=b）的个数/y=b的个数

    #初始化F字典，用于存储输入对象属于各个类别的概率 key：类别标签，value：概率
    F = {}

    # 计算输入对象属于各个类别的概率
    for y in p_y:
        F[y] = p_y[y]
        for j in range(len(input_feature)):
            # P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
            F[y] = F[y]*P_xy['feature'+ str(j+1)+'='+str(input_feature[j]) + '|y=' + str(y)]
    # feature_label=概率最大值对应的类别
    features_label = max(F, key=F.get)

    #打印结果
    print('input feature:',str(input_feature))
    print('posibility:',F)
    print ('classified label:',features_label)
    #返回预测类标
    return (features_label)

#计算错误率函数
def mis(test_index,test_data,train_data,train_index,label):
    #初始化miscount，用于计算错误预测次数
    miscount = 0

    #对每一个测试用例进行bayes分类预测
    for i in range(len(test_data)):
        result = bayes(label,test_data[i],train_data,train_index)
        #对预测结果进行统计
        if result == label[test_index[i]]:
            continue
        else:
            miscount += 1

    # 计算错误率=错误预测数/总数*100%
    s=len(test_data)
    misratio = miscount/s*100

    #返回错误率
    return misratio

#读入文件
filename = 'titanic.dat'
data_set, label, feature1, feature2, feature3 = data_preprocess(filename)

#找每个属性的分点
point_entropy1, split_point1 = find_split_piont(feature1)
point_entropy2, split_point2 = find_split_piont(feature2)
point_entropy3, split_point3 = find_split_piont(feature3)

#对每个属性取值进行离散化
split_feature1 = split_feature(split_point1,feature1)
split_feature2 = split_feature(split_point2,feature2)
split_feature3 = split_feature(split_point3,feature3)

#将离散化后取值更新到数据集
data_set = np.array(data_set,dtype=float)

for i in range(len(data_set)):
    data_set[i][0] = split_feature1[i]
    data_set[i][1] = split_feature2[i]
    data_set[i][2] = split_feature3[i]

#划分测试集和训练集
testsize=0.3
train_data,test_data,train_index,test_index = splict_test_and_train(data_set,testsize)

#计算准确率
mis_ratio=mis(test_index,test_data,train_data,train_index,label)
accuracy=100-mis_ratio

#输出最后结果
print "The accuracy on test data: %f " %accuracy
