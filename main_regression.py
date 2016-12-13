import pandas as pd
import numpy as np
import warnings
import sys
from visual import copy_pics_into_folders, delete_folders
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble
warnings.filterwarnings('ignore')

# coding: utf-8

__author__ = 'Ming Li'

'''
This application forms a submission from Ming Li
in regards to leaf convnet challenge on Kaggle community
'''

# set display right
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)


# load raw data

test = pd.read_csv('leaf/test.csv')
train = pd.read_csv('leaf/train.csv')


# picking useful data points

regressors = train.select_dtypes(exclude=(np.int8, np.int64, np.object)).copy()
regressand = train.select_dtypes(exclude=(np.int8, np.int64, np.float)).copy()

# codifying types of species

regressand['species_id'] = pd.Categorical.from_array(regressand['species']).codes
mapping = regressand[['species_id','species']].set_index('species_id').to_dict()['species']
regressand.drop('species', axis=1, inplace=True)
regressand = np.ravel(regressand)

# model generalization

kf_generator = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

# feature scaling using standard deviation as denominator

regressors_std = regressors.\
    apply(preprocessing.scale, with_mean=False, with_std=True, axis=0)

# Logistic regression

regressors_std = np.column_stack(
    (np.ones(regressors.shape[0]), regressors_std)
)  # add constant
regressors = np.column_stack(
    (np.ones(regressors.shape[0]), regressors)
)  # add constant

# # hold out
#
# x_train, x_test, y_train, y_test = model_selection.\
#     train_test_split(regressors_std, regressand, test_size=.3, random_state=1)

# fit training set

reg = linear_model.LogisticRegression(fit_intercept=False)
# regressors already contains manually added intercept
# reg.fit(regressors_std, regressand)

reg = ensemble.AdaBoostClassifier(base_estimator=reg, n_estimators=50, learning_rate=1, algorithm='SAMME.R',
                                  random_state=None)
reg.fit(regressors_std, regressand)


print('Using given features by Kaggle, Logistic Regression model accuracy is: ', end='')

avg_scores = model_selection.cross_val_score(
    reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)

print('{0:.2f}%'.format(100 * np.mean(avg_scores)), flush=True, end='\n')

# combine train and test

test['species'] = np.nan  # adding species to testing set
test = test[train.columns]  # only keep columns in training

combined = pd.concat(objs=[test, train], axis=0).sort_values('id', ascending=True)
regressors = combined.select_dtypes(exclude=(np.int8, np.int64, np.object)).copy()
regressors_std = regressors.apply(preprocessing.scale, axis=0)  # using standard deviation as denominator
regressors_std = np.column_stack((np.ones(regressors_std.shape[0]), regressors_std))  # add constant

# moving classified pictures into folders

combined['species_id'] = reg.predict(regressors_std)  # making predictions
combined['species_predicted'] = combined['species_id'].map(mapping)
result = combined.select_dtypes(include=(np.int8, np.int64, np.object)).copy()
table = result[['species_predicted', 'id']].sort_values(by=['species_predicted', 'id'], ascending=True)\
    .set_index('species_predicted')

copy_pics_into_folders(table)


key = input('continue to produce submission?\n')

if str(key).lower() != 'yes':
    delete_folders(table)
    sys.exit()

else:
    # produce submission.csv as requested by kaggle
    print('producing submission...', flush=True, end='')
    probs = reg.predict_proba(regressors_std)
    probs_df = pd.DataFrame(probs, columns=mapping.values(), index=combined['id'])
    submission = probs_df.loc[test['id']]
    submission.to_csv('submission.csv', encoding='utf-8', header=True)
    delete_folders(table)
    print('done', flush=True)
    sys.exit()
