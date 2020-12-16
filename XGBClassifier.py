
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import tpe

import os

import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# initialize the environment
# import janestreet
# env = janestreet.make_env()
# iter_test = env.iter_test()


print('Loading training data...')
train = pd.read_csv('train.csv')
print('Finished.')

print(train.head())
print(train.columns)
# preprocessing
print('Preprocessing...')
train = train[train['weight'] != 0]  # do not train data with 0 weight
train['action'] = (train['resp'].values > 0).astype('int')  # only work with positive return expected

# feature Engineering
X = train.loc[:, train.columns.str.contains('feature')]
f_mean = X.mean()
X.fillna(f_mean)  # fill na values with feature mean

y = train.loc[:, 'action']
print('Finished.')

# split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def main():
    # create model
    print('Creating classifier...')
    clf = xgb.XGBClassifier(
        colsample_bytree=0.6000000000000001,
        gamma=0.55,
        learning_rate=0.1,
        max_depth=5,
        min_child=24.0,
        n_estimators=40,
        subsample=0.802038259450567,
        tree_method='gpu_hist'
    )
    print('Finished.')

    # train classifier
    print('Training classifier...')
    clf.fit(X_train, y_train)
    print('Finished.')

    # score classifeir
    print('Scoring model...')
    y_pred = clf.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


# for (test_df, sample_prediction_df) in iter_test:
# X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
# y_preds = clf.predict(X_test)
# sample_prediction_df.action = y_preds
# env.predict(sample_prediction_df)

if __name__ == '__main__':
    main()
