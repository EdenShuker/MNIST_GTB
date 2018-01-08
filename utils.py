import random
from time import time
import mnist
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score


def main1():
    # TODO - note, need to install python-mnist package
    # 'data' is the name of the directory with the unzipped files inside
    mndata = mnist.MNIST('data')
    images, labels = mndata.load_testing()
    # images is a list, each item in it is a list of bytes, i.e. each is a number of 0-255
    # labels is a list (array-type) of the desired label

    # example of accessing data
    index = random.randrange(0, len(images))  # choose an index
    print(mndata.display(images[index]))
    print labels[index]  # get the label of the ith image


def main2():
    mndata = mnist.MNIST('data')
    images, labels = mndata.load_training()
    d_mat = xgb.DMatrix(images, labels)
    params = {}
    res = xgb.cv(params, d_mat)
    print res


def main3():
    # load train data
    mndata = mnist.MNIST('data')
    train, train_labels = mndata.load_training()
    train, train_labels = np.array(train), np.array(train_labels)  # as numpy
    print 'loaded data'

    # train a model
    model = xgb.XGBClassifier(n_estimators=10)
    model.fit(train, train_labels)
    print model  # just for checking on it

    # predict on test
    test, test_labels = mndata.load_testing()
    test, test_labels = np.array(test), np.array(test_labels)
    preds = model.predict(test)
    predictions = [round(val) for val in preds]

    # accuracy
    acc = accuracy_score(test_labels, predictions)
    print 'accuracy %0.2f%%' % (acc * 100.0)


def main4():
    def data_as_np(data_and_labels):
        data, labels = data_and_labels
        data, labels = np.array(data), np.array(labels)
        return xgb.DMatrix(data, labels)

    mndata = mnist.MNIST('data')
    train_dmat = data_as_np(mndata.load_training())

    params = {'eta': 0.1, 'seed': 0, 'max_depth': 3, 'num_class': 10}
    res = xgb.cv(params, train_dmat, num_boost_round=10, nfold=3, metrics=['merror'])
    # print res
    print res
    print 'accuracy:', 1 - res['test-merror-mean'][-1]


if __name__ == '__main__':
    t = time()
    print 'start'

    # main1()
    # main2()
    main3()
    # main4()

print 'time for running all:', time() - t
