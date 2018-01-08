from time import time

import numpy as np
import xgboost as xgb
import mnist
from sklearn.metrics import accuracy_score


def data_as_np(data_and_labels):
    data, labels = data_and_labels
    data, labels = np.array(data), np.array(labels)
    return data, labels


def model_fit(model, data_and_labels, test_d_l, use_train_cv=True, cv_folds=3, early_stopping_rounds=3):
    data, labels = data_and_labels
    params = {'eta': 0.1, 'seed': 0, 'max_depth': 3, 'num_class': 10}
    if use_train_cv:
        model_params = model.get_xgb_params()
        model_params['num_class'] = 10
        cvresult = xgb.cv(params,
                          xgb.DMatrix(data, label=labels),
                          num_boost_round=model.get_params()['n_estimators'],
                          nfold=cv_folds,
                          stratified=True,
                          metrics=['mlogloss'],
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                     xgb.callback.early_stop(3)])
        print cvresult
        # TODO the code from source produced error, says that cvresult is dict and has no shape-param
        model.set_params(n_estimators=len(cvresult['test-mlogloss-mean']))

    # Fit the algorithm
    model.fit(data, labels, eval_metric='mlogloss')

    print model, '\n'
    test_data, test_labels = test_d_l
    preds = model.predict(test_data)
    predictions = [round(val) for val in preds]
    acc = accuracy_score(test_labels, predictions)
    print 'accuracy %0.2f%%' % (acc * 100.0)


def main():
    model = xgb.XGBClassifier(learning_rate=0.1,
                              n_estimators=3,
                              max_depth=3,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              scale_pos_weight=1,
                              objective='multi:softprob',
                              seed=27)
    mndata = mnist.MNIST('data')
    train_data_and_labels = data_as_np(mndata.load_training())
    test_data_and_labels = data_as_np(mndata.load_testing())
    model_fit(model, train_data_and_labels, test_data_and_labels)


if __name__ == '__main__':
    t0 = time()
    print 'start'

    main()

    print 'time to run all:', time() - t0
