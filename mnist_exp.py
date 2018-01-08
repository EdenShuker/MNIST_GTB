from time import time

import numpy as np
import xgboost as xgb
import mnist
from sklearn.metrics import accuracy_score
import pickle


def data_as_np(data_and_labels):
    data, labels = data_and_labels
    data, labels = np.array(data), np.array(labels)
    return data, labels


class MnistModel(object):
    model_filename = 'xgb.model'

    def __init__(self, model=None, lr=0.1, n_estimators=10, max_depth=5, min_child_weight=1,
                 gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=27):
        if model is not None:
            self.model = model
        else:
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
        mndata = mnist.MNIST(dir_name)
        train_data, train_labels = data_as_np(mndata.load_training())

        if with_cv:
            self._use_cv(train_data, train_labels, cv_folds, early_stopping_rounds)

        self.model.fit(train_data, train_labels, eval_metric='mlogloss')
        print self.model, '\n'

        test_data, test_labels = data_as_np(mndata.load_testing())
        self.predict_and_check_accuracy(test_data, test_labels)

        if with_saving:
            pickle.dump(self.model, MnistModel.model_filename)

    def predict_and_check_accuracy(self, data, labels):
        preds = self.model.predict(data)
        predictions = [round(val) for val in preds]
        acc = accuracy_score(labels, predictions)
        print 'accuracy %0.2f%%' % (acc * 100.0)

    @staticmethod
    def load():
        model = pickle.load(MnistModel.model_filename)
        return MnistModel(model=model)


def train_model():
    model = MnistModel()
    model.train_on('data')


def load_and_check_accuracy():
    mndata = mnist.MNIST('data')
    test_data, test_labels = data_as_np(mndata.load_testing())

    model = MnistModel.load()
    model.predict_and_check_accuracy(test_data, test_labels)


if __name__ == '__main__':
    t0 = time()
    print 'start'

    train_model()

    print 'time to run all:', time() - t0
