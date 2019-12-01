# -*- coding: UTF-8 -*-
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def distance(self, featureA, featureB):
        # calcualte the Euclidean distance of two samples
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # input is an array of features and labels
        self.train_data = np.array(X, dtype=float)
        self.train_label = np.array(y, dtype=float)
        None

    def predict(self, X):
        # Return array of predictions where there is one prediction for each set of features
        test_result = []

        # each time predict one test sample
        for test_item in X:

            # calculate the distance between test sample and all the train samples
            distance = []
            for train_item in self.train_data:
                one_differs = self.distance(test_item, train_item)
                distance.append(one_differs)
            # argsort() return the index of elements sorted from small to large
            rank = np.argsort(distance)
            vote_label = []

            # get the the label of k elements have smallest distance with test sample
            for i in range(self.k):
                vote_label.append(self.train_label[rank[i]])
            # count the vote
            one_count = 0
            zero_count = 0
            for j in vote_label:
                if j == 1:
                    one_count += 1
                else:
                    zero_count += 1
            if one_count > zero_count:
                result = 1
            else:
                result = 0

            # append this prediction to test_result[]
            test_result.append(result)

        # test_result contains all the predictions
        test_result = np.array(test_result)

        return test_result


class ID3:
    def __init__(self, nbins, data_range):
        # Decision tree state here
        self.bin_size = nbins
        # data_range (array1[min_col1,...,min_col30],array2[max_col1,...,max_col30])
        self.range = data_range
        # trace records the prediction path
        self.trace = []

    def preprocess(self, data):
        '''
        function : normalize the data
        :param data: input data
        :return:  normalized data
        '''
        # dataset only has continuous data
        # def clip(a, a_min, a_max, out=None). rescale the data into (0,1)
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        # np.floor() return the the interger smallar than input
        # all data in [0,3]
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def subDataset(self, origin_data, feature_i, value):
        '''
        function : select the data have the same chosen feature value with training sample
        :param origin_data: original data set
        :param feature_i: the column(feature) that gonna be the splitting point
        :param value: the value of chosen feature of training sample
        :return: selected sub data set without this column(feature)
        '''
        # store sub data set
        subDataset = []
        for train_item in origin_data:
            # choose samples have same chosen feature value
            if train_item[feature_i] == value:
                # delete the chosen column(feature)
                train_item = np.delete(train_item, feature_i, 0)
                # add this sample to sub data set
                subDataset.append(train_item)

        return subDataset

    def bestGainFeature(self, dataset):
        '''
        function : select the best feature with highest information gain
        :param dataset: the dataset for analysis
        :return: the index of the best feature
        '''
        # turn into array
        dataset = np.array(dataset)
        # the number of training samples
        num_train = dataset.shape[0]
        # the number of features
        num_feature = dataset.shape[1] - 1
        # store all the entropy of every feature
        entropy = []

        # calculate the entropy of every feature
        for feature_i in range(num_feature):
            feature_count = {}
            # count every training sample's info in feature_count : feature-value and label
            for train_item in range(num_train):
                feature_value = dataset[train_item][feature_i]
                feature_label = dataset[train_item][-1]
                if feature_value not in feature_count.keys():
                    feature_count[feature_value] = [0, 0]
                if feature_label == 0:
                    feature_count[feature_value][0] += 1
                if feature_label == 1:
                    feature_count[feature_value][1] += 1

            # calculate the info of this feature
            entropy_i = 0
            # calculate every sub-entropy, based on different value of the same feature
            for value in feature_count.keys():

                # the number of class 0
                count0 = feature_count[value][0]
                # the number of class 1
                count1 = feature_count[value][1]
                # total number of samples have this feature value
                count_all = count0 + count1
                # proportion of class 0
                fre0 = count0 / count_all
                # proportion of class 1
                fre1 = count1 / count_all

                # avoid edge case : fre0 or fre1 is zero
                sub_entropy0, sub_entropy1 = 0, 0
                if fre0 != 0:
                    sub_entropy0 = (-1) * fre0 * (np.log2(fre0))
                if fre1 != 0:
                    sub_entropy1 = (-1) * fre1 * (np.log2(fre1))

                # entropy formula
                entropy_i += count_all / num_train * (sub_entropy0 + sub_entropy1)
            # store the entropy of this feature
            entropy.append(entropy_i)

        # calculate the entropy of the whole dataset
        sub_entropy0, sub_entropy1 = 0, 0
        fre0 = np.sum(dataset[:, -1] == 0) / num_train
        fre1 = np.sum(dataset[:, -1] == 1) / num_train
        if fre0 != 0:
            sub_entropy0 = (-1) * fre0 * (np.log2(fre0))
        if fre1 != 0:
            sub_entropy1 = (-1) * fre1 * (np.log2(fre1))
        info_train = sub_entropy0 + sub_entropy1

        # calculate the information gain
        gain = info_train - entropy
        # get the feature with highest information gain
        best_feature_index = np.argmax(gain)

        return best_feature_index

    def voteClass(self, trace, dataset, test_item):
        '''
        function : when prediction cannot move on(testing sample have the feature value that all training samples don't have)
                    choose the class that have higher proportion when reach the last predicting step
        :param trace: a list stores all the chosen features in the prediction path in order
        :param dataset: the sub-dataset in the last predicting step
        :param test_item: testing sample that gonna be predicted
        :return: class label -- as the prediction result
        '''
        subdataset = dataset
        # store the original feature name in list
        feature_list = list(range(len(subdataset[0]) - 1))

        # reach the last predicting step
        for feature in trace[:-1]:
            value = test_item[feature]
            # get the index of the chosen feature -- the corresponding column in the sub-dataset
            feature_to_index = feature_list.index(feature)
            # select training samples that have same chosen feature value with testing sample
            subdataset = self.subDataset(subdataset, feature_to_index, value)
            # delect the feature_name from feature_list so that we can map the column to feature_name in future
            del feature_list[feature_to_index]

        # get the label of all the selected training samples
        label_list = [train_item[-1] for train_item in subdataset]
        # use labelStatistics get the class that has more training samples
        vote_res = self.labelStatistics(label_list)

        return vote_res



    def labelStatistics(self, label_list):
        '''
        function : get the label that has more training samples
        :param label_list: a list stores the labels of samples
        :return: class label
        '''
        label_count = {}
        for label_i in label_list:
            label_count[label_i] = label_count.get(label_i, 0) + 1
        label_count = sorted(label_count.items(), key=lambda item: item[1])
        return label_count[-1][0]


    def buildTree(self, train_data, feature_list):
        '''
        function: create the decision tree
        :param train_data: training data set
        :param feature_list: a list with all feature's name (here simply it as [0,1,...,29])
        :return: decision tree (in the form of dictionary)
        '''
        # label_list contains all the label of the training samples
        label_list = [train_item[-1] for train_item in train_data]

        # if dataset only contains one class, then return this class
        if label_list.count(label_list[0]) == len(label_list):
            return label_list[0]
        # if all the features are used as splitting node, then return the class have more samples
        if len(train_data[0]) == 1:
            return self.labelStatistics(label_list)

        # find the best column (feature) to spilt the dataset
        best_f = self.bestGainFeature(train_data)
        # get the feature name of this column
        best_f_label = feature_list[best_f]

        # initialize the tree
        tree = {best_f_label: {}}
        # delete this feature name from feature_list as it has been chosen
        del (feature_list[best_f])
        # get all value of this best feature
        best_f_value = [example[best_f] for example in train_data]
        f_value_set = set(best_f_value)

        # build sub-tree of different value of this feature
        for value in f_value_set:
            # get the subdataset that have same chosen feature value
            subDataset = self.subDataset(train_data, best_f, value)
            # update the feature_list by deleting this feature's name
            subLabels = feature_list[:]
            # build sub-tree underneath this tree structure recursively
            tree[best_f_label][value] = self.buildTree(subDataset, subLabels)

        return tree

    def train(self, X, y):
        '''
        function : build the decision tree with input training data
        :param X: training data features
        :param y: training data labels
        :return: decision tree
        '''
        # normalize the training data
        normX = self.preprocess(X)
        y = np.array([y])
        # turn data into [[feature1,...,feature30,label], [...],[...]...]
        categorical_data = np.concatenate((normX, y.T), axis=1)
        self.norm_xy = categorical_data
        num_feature = categorical_data.shape[1] - 1
        # store the original feature name in list
        feature_list = list(range(0, num_feature))
        # build tree
        self.decision_tree = self.buildTree(categorical_data, feature_list)

    def classify(self, test_item, tree=None):
        '''
        function: predict the label of one input testing sample
        :param test_item: input testing sample feature
        :param tree: decision tree of sub-decision tree
        :return: predicted label
        '''
        # get the name of first feature in the tree
        f_index = list(tree.keys())
        f_index = f_index[0]
        # get the sub-tree below this feature eg.{0:{},1:{},..}
        next_dict = tree[f_index]
        # record the path
        self.trace.append(f_index)

        # set the predict_label = -1 to avoid edge case: tree doesn't contain sample's some feature values
        predict_label = -1
        # choose the path that this sample satisfies : key that have the same feature value with sample
        for f_index_value in next_dict.keys():
            if test_item[f_index] == f_index_value:
                # if there is next level, move to next level of tree
                if type(next_dict[f_index_value]).__name__ == 'dict':
                    predict_label = self.classify(test_item, next_dict[f_index_value])
                # else, reach the leaf, return the label of the leaf
                else:
                    predict_label = next_dict[f_index_value]

        return predict_label

    def predict(self, X, tree=None):
        '''
        function: make prediction for the input testing data samples
        :param X: input testing data samples features
        :param tree: decision tree
        :return: array of predictions where there is one prediction for each set of features
        '''
        # normalize the data
        categorical_data = self.preprocess(X)
        # store the prediction
        test_result = []

        # predict every sample in the dataset
        for test_item in categorical_data:
            # initialize the trace
            self.trace = []
            # prediction
            predict_label = self.classify(test_item, self.decision_tree)
            # meet edge case: tree doesn't contain sample's some feature values
            if predict_label == -1:
                # use voteClass() to choose class with more samples in the last step
                predict_label = self.voteClass(self.trace, self.norm_xy, test_item)
            # store this prediction
            test_result.append(predict_label)

        # turn predictions into array and return
        test_result = np.array(test_result)
        return test_result


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    # shuffle the order of input sample
    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        # input is array of features and labels
        # train the whole data set for steps//y.size time
        for _ in range(steps // y.size):
            # everytime finish training all train data, shuffle the train data set
            X, y = self.shuffle(X, y)
            predictions = self.predict(X)
            # update the weight and b after training one train sample
            for i in range(len(predictions)):
                self.w += self.lr * (y[i] - predictions[i]) * X[i]
                self.b += self.lr * (y[i] - predictions[i])

    def predict(self, X):
        # Return array of predictions where there is one prediction for each set of features
        activations = np.zeros(len(X))
        # prediction the label of each test item
        for i in range(len(X)):
            # activation formula
            fx = np.dot(X[i], self.w) + self.b
            if fx > 0:
                activations[i] = 1
            else:
                activations[i] = 0
        return activations


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    # calculate the error
    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    # calculate the gradient of the output layer
    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    # shuffle the order of input sample
    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    # train the network
    def train(self, X, y, steps):
        for s in range(steps):
            # everytime finish training all the samples once, then shuffle
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            # expand_dims(input vetor, axis), when axis=0, add 1 dimension in the row, [1,2]->[[1,2]]; axis = 1,
            # add 1 dimension in column[1,2]->[[1],[2]]
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            # forward path
            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

            # backward error propagation
            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        # ravel() turn pred into 1 dimension
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # forward pass
        self.input = input
        # calculate the output
        self.output = np.dot(input, self.w) + self.b
        return self.output

    def backward(self, gradients):
        # Write backward pass here
        # update weights
        self.w -= np.dot(self.input.T, gradients) * self.lr
        # update b
        self.b -= self.lr * gradients
        return np.dot(gradients, self.w.T)


class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        # forward pass
        # calculate the output
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, gradients):
        # backward pass
        # calculate the gradient
        return gradients * self.output * (1 - self.output)
