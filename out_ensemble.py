import numpy as np
from utilities import extract, batch_iter, generate_training_set
from sklearn import metrics, linear_model, model_selection, preprocessing, ensemble, svm, naive_bayes, neighbors
import sys
import pandas as pd

# parameters

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

train, label, data = extract(INPUT_PATH + 'train.csv', target='species')

regressors = train.apply(preprocessing.scale, axis=0, with_mean=True, with_std=True)
regressand = data['species']

# KFold generator
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

# Logistics Regression
# regressors = np.column_stack((np.ones(len(regressors)), regressors))

clf = linear_model.LogisticRegression(fit_intercept=True)

average_score = model_selection.cross_val_score(clf, regressors, regressand, scoring='accuracy', cv=kf)
print('Using given features by Kaggle, Logistic Regression model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(average_score), average_score), flush=True, end='\n')

