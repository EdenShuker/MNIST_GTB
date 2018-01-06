import random
from time import time
import mnist
import xgboost as xgb


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


if __name__ == '__main__':
    t = time()
    print 'start'

    # main1()
    main2()

    print 'time for running all:', time() - t
