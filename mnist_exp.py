from time import time

import numpy as np
import xgboost as xgb
import mnist
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix


def data_as_np(data_and_labels):
    """
    :param data_and_labels: tuple of (data, labels)
    :return: the tuple where each object is np-array
    """
    data, labels = data_and_labels
    data, labels = np.array(data), np.array(labels)
    return data, labels


class MnistModel(object):
    model_filename = 'xgb.model'

    def __init__(self, model=None, lr=0.1, n_estimators=30, max_depth=5, min_child_weight=1,
                 gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=27):
        if model is not None:  # from given model
            self.model = model
        else:  # create new model
            self.model = xgb.XGBClassifier(learning_rate=lr,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_child_weight=min_child_weight,
                                           gamma=gamma,
                                           subsample=subsample,
                                           colsample_bytree=colsample_bytree,
                                           scale_pos_weight=scale_pos_weight,
                                           objective='multi:softprob',
                                           seed=seed)

    def _use_cv(self, data, labels, cv_folds, early_stopping_rounds):
        """ run cv with the parameters below to find the best n_estimators for the model """
        params = {'eta': 0.1, 'seed': 0, 'max_depth': self.model.get_params()['max_depth'], 'num_class': 10}
        cvresult = xgb.cv(params,
                          xgb.DMatrix(data, label=labels),
                          num_boost_round=self.model.get_params()['n_estimators'],
                          nfold=cv_folds,
                          stratified=True,
                          metrics=['mlogloss'],
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                     xgb.callback.early_stop(3)])
        print cvresult
        self.model.set_params(n_estimators=len(cvresult['test-mlogloss-mean']))

    def train_on(self, dir_name, with_saving=True, with_cv=False, cv_folds=3, early_stopping_rounds=3):
        # load data
        mndata = mnist.MNIST(dir_name)
        train_data, train_labels = data_as_np(mndata.load_training())

        if with_cv:
            self._use_cv(train_data, train_labels, cv_folds, early_stopping_rounds)

        # train the model
        self.model.fit(train_data, train_labels, eval_metric='mlogloss')
        print self.model, '\n'

        # check the accuracy on the test-set
        test_data, test_labels = data_as_np(mndata.load_testing())
        self.predict_and_check_accuracy(test_data, test_labels)

        if with_saving:
            pickle.dump(self.model, open(MnistModel.model_filename, 'wb'))

    def predict_and_check_accuracy(self, data, labels):
        # predict on test-set
        preds = self.model.predict(data)
        predictions = [round(val) for val in preds]

        # find accuracy
        acc = accuracy_score(labels, predictions)
        print 'accuracy %0.2f%%' % (acc * 100.0)

        # confusion matrix
        print confusion_matrix(labels, predictions)


    @staticmethod
    def load(filename=None):
        """ load a model from file """
        if filename is None:
            filename = MnistModel.model_filename
        model = pickle.load(open(filename, 'rb'))
        return MnistModel(model=model)

    def predict_on(self, x):
        preds = self.model.predict(x)
        return [round(val) for val in preds]


def train_model():
    model = MnistModel()
    model.train_on('data')


def load_and_check_accuracy(filename):
    # load data
    mndata = mnist.MNIST('data')
    test_data, test_labels = data_as_np(mndata.load_testing())

    # load model from file and check accuracy on the test-set
    model = MnistModel.load(filename)
    model.predict_and_check_accuracy(test_data, test_labels)

    # with visual example
    print '\n'
    # choose up to
    samples_indexes = np.random.randint(len(test_labels), size=5)
    samples_data = [test_data[i] for i in samples_indexes]
    samples_labels = [test_labels[i] for i in samples_indexes]
    preds = model.predict_on(samples_data)
    # print results
    for sample, label, pred_label in zip(samples_data, samples_labels, preds):
        print mndata.display(sample)
        print 'gold:', label, ', pred:', pred_label, '\n'


if __name__ == '__main__':
    t0 = time()
    print 'start'

    # train_model()
    load_and_check_accuracy('xgb_30.model')

    print 'time to run all:', time() - t0
