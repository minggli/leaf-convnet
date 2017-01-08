import numpy as np
from utilities import extract, batch_iter, generate_training_set
from sklearn import metrics, linear_model, model_selection, preprocessing, ensemble, svm, naive_bayes, neighbors
import sys
import pandas as pd
from utilities import delete_folders, move_classified

# parameters

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

train, label, data = extract(INPUT_PATH + 'train.csv', target='species')
_, _, test = extract(INPUT_PATH + 'test.csv')

val_prob = list()
cnn_result = pd.read_csv('0.03074.csv', index_col='id', encoding='utf-8')
val_prob.append(np.array(cnn_result))

scaler = preprocessing.StandardScaler(with_std=True, with_mean=False)


regressors = scaler.fit(train).transform(train)
regressand = data['species']

test_regressors = scaler.fit(test).transform(test)

# KFold generator
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

clf = linear_model.LogisticRegression(fit_intercept=True)

average_score = model_selection.cross_val_score(clf, regressors, regressand, scoring='neg_log_loss', cv=kf)
print('Using given features by Kaggle, Logistic Regression model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(average_score), average_score), flush=True, end='\n')

clf.fit(regressors, regressand)
pred = clf.predict_proba(test_regressors)
val_prob.append(pred)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)

average_score = model_selection.cross_val_score(clf, regressors, regressand, scoring='neg_log_loss', cv=kf)
print('Using given features by Kaggle, K-Nearest Neighbour model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(average_score), average_score), flush=True, end='\n')

clf.fit(regressors, regressand)
pred = clf.predict_proba(test_regressors)
val_prob.append(pred)
#
# clf = svm.SVC(probability=True)
#
# average_score = model_selection.cross_val_score(clf, regressors, regressand, scoring='neg_log_loss', cv=kf)
# print('Using given features by Kaggle, SVM model accuracy is: ', end='')
# print('{1} averaging in {0:.2f}%'.format(100 * np.mean(average_score), average_score), flush=True, end='\n')
#
# clf.fit(regressors, regressand)
# pred = clf.predict_proba(test_regressors)
# val_prob.append(pred)

clf = ensemble.RandomForestClassifier(n_estimators=280, max_depth=None, max_leaf_nodes=None)

average_score = model_selection.cross_val_score(clf, regressors, regressand, scoring='neg_log_loss', cv=kf)
print('Using given features by Kaggle, Random Forest model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(average_score), average_score), flush=True, end='\n')

clf.fit(regressors, regressand)
pred = clf.predict_proba(test_regressors)
val_prob.append(pred)

output = np.around(np.mean(val_prob, axis=0), decimals=1)
print(output)


def submit(raw):

    delete_folders()

    move_classified(test_data=raw, train_data=data, columns=label.columns, index=test.index, path=IMAGE_PATH)

    df = pd.DataFrame(data=raw, columns=label.columns, index=test.index)
    df.to_csv('submission.csv', encoding='utf-8', header=True, index=True)

submit(output)

