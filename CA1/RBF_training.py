# -*- coding: utf-8 -*-
# @author:ZHOU YICHEN
import numpy as np
from RBF_classifier import RBF
import scipy.io as sio
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from visualization import plot_cvscores_RBF, plot_decision

# load data
matfn=u'datasets.mat'
data=sio.loadmat(matfn)
data_train = data['data_train']
label_train = data['label_train']
data_test = data['data_test']
# Feature scaling
scaler1 = MinMaxScaler().fit(data_train)
scaler2 = MinMaxScaler().fit(data_test)
X_train, X_test = scaler1.transform(data_train), scaler2.transform(data_test)
# define params
params = {'num_neurons': list(np.linspace(10, 100, 10).astype(int)),
          "gamma": list(np.linspace(0.01, 1, 10))}
# use GridsearchCV to find the best params
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
grid_search = GridSearchCV(RBF(), params, cv=skf, return_train_score=True)
grid_result = grid_search.fit(X_train, label_train)
scores = grid_search.cv_results_['mean_test_score'].reshape(10, 10)
plot_cvscores_RBF(scores, params)
print(skf)
# produce the best model
model = grid_search.best_estimator_
# KFold cross validation
kfold = KFold(n_splits=4, shuffle=False)
acc_train = 0
acc_valid = 0
for train_index, valid_index in kfold.split(label_train):  # 4-fold

    model.fit(X_train[train_index], label_train[train_index])
    res_train = model.predict(X_train[train_index])
    res_valid = model.predict(X_train[valid_index])
    # print(accuracy_score(res_train, label_train[train_index]))
    # print(accuracy_score(res_valid, label_train[valid_index]))
    acc_train += float(sum(res_train == label_train[train_index]) / len(train_index))
    acc_valid += float(sum(res_valid == label_train[valid_index]) / len(valid_index))

print('ACC using SVM: ' + str(acc_train / 4) + ' | ' + 'Valid: ' + str(acc_valid / 4))

# predict the label of testing set
y_hat = model.predict(X_test)

print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(y_hat.reshape(1,-1))
plot_decision(data_train, label_train, model)
