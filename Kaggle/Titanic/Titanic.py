#!/usr/bin/env
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
import csv

from sklearn.model_selection import train_test_split


def load_data(is_train, fileCsv):
    data = pd.read_csv(fileCsv)

    # 性别
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    # data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'U': 0}).astype(int)
    # print data['Embarked']
    embarked_data = pd.get_dummies(data.Embarked)
    # print embarked_data
    # embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    if is_train:
        y = data['Survived']
        return x, y
    else:
        return x, data['PassengerId']


def write_result(model, type):
    x, passenger_id = load_data(False, fileCsv='./input/Titanic.test.csv')

    if type == 3:
        x = xgb.DMatrix(x)
    y = model.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0

    predictions_file = open("Prediction_%d.csv" % type, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(passenger_id, y))
    predictions_file.close()


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    # print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate


if __name__ == "__main__":
    x, y = load_data(True, fileCsv="./input/Titanic.train.csv")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=1)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
    # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
    bst = xgb.train(param, data_train, num_boost_round=30, evals=watch_list)
    y_hat = bst.predict(data_test)
    write_result(bst, 3)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')
    print 'XGBoost：%.3f%%' % xgb_rate
