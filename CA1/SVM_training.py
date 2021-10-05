# -*- coding: utf-8 -*-
# @author:ZHOU YICHEN
import scipy.io as sio
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from visualization import plot_cvscores_SVM, plot_tsne, plot_decision, plot_leaning
from sklearn.metrics import accuracy_score

# load data
matfn = u'datasets.mat'
data = sio.loadmat(matfn)
data_train = data['data_train']
label_train = data['label_train']
data_test = data['data_test']
plot_tsne(data_train, label_train.ravel())

# Feature scaling
scaler1 = MinMaxScaler().fit(data_train)
scaler2 = MinMaxScaler().fit(data_test)
X_train, X_test = scaler1.transform(data_train), scaler2.transform(data_test)

# define the type of SVM kernel: rbf(Gaussian kernel function) and range of C and gamma
params = {'kernel': ['rbf'],'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# search the best params in SVM
grid_search = GridSearchCV(SVC(), params, cv=5, return_train_score=True)
grid_result = grid_search.fit(X_train, label_train.ravel())

# save the GridsearchCV results and plot the scores of params
results = pd.DataFrame(grid_search.cv_results_)
scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)
plot_cvscores_SVM(scores, params)

# get the best params and produce the best model
model = grid_search.best_estimator_

# KFold cross validation
kfoldscores = cross_val_score(estimator=model, X=X_train, y=label_train.ravel(), cv=10, n_jobs=1)
print(kfoldscores)

kfold = KFold(n_splits=4, shuffle=False)
acc_train = 0
acc_valid = 0
for train_list, valid_list in kfold.split(label_train):  # 4-fold

    model.fit(X_train[train_list], label_train[train_list])

    res_train = model.predict(X_train[train_list])
    res_valid = model.predict(X_train[valid_list])
    print(accuracy_score(res_train, label_train[train_list]))
    print(accuracy_score(res_valid, label_train[valid_list]))
    acc_train += float(sum(res_train == label_train[train_list].squeeze()) / len(train_list))
    acc_valid += float(sum(res_valid == label_train[valid_list].squeeze()) / len(valid_list))

print('ACC using SVM: ' + str(acc_train / 4) + ' | ' + 'Valid: ' + str(acc_valid / 4))

# predict the label of testing set
print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
y_hat = model.predict(X_test)
plot_tsne(X_test, y_hat.ravel())
print(y_hat)
plot_decision(data_train, label_train, model)
