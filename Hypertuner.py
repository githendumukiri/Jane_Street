import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

# load data
print('Loading training data...')
train = pd.read_csv('train.csv')
print('Finished.')

# preprocessing
print('Preprocessing')
train = train[train['weight'] != 0]  # do not train data with 0 weight
train['action'] = (train['resp'].values > 0).astype('int')  # only work with positive return expected

# feature Engineering
X = train.loc[:, train.columns.str.contains('feature')]
f_mean = X.mean()
X.fillna(f_mean)  # fill na values with feature mean

y = train.loc[:, 'action']
print('Finished.')

print('Split Data...')
# split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('Finished.')


def main():
    space = {
        'max_depth': hp.choice('max_depth', np.arange(10, 20, dtype=int)),
        'min_child_weight': hp.quniform('min_child', 1, 30, 1),
        'subsample': hp.uniform('subsample', 0.8, 1),
        'n_estimators': hp.choice('n_estimators', np.arange(100, 10000, 100, dtype=int)),
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
    }

    trials = Trials()

    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)

    print(best)


def hyperparameter_tuning(space):
    print('Building Model...')
    model = xgb.XGBClassifier(
        n_estimators=space['n_estimators'],
        max_depth=space['max_depth'],
        min_child_weight=space['min_child_weight'],
        random_state=42,
        subsample=space['subsample'],
        learning_rate=space['learning_rate'],
        gamma=space['gamma'],
        colsample_bytree=space['colsample_bytree'],
        tree_method='gpu_hist'
        )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    model.fit(X_train, y_train,
              eval_set=evaluation, eval_metric="rmse",
              early_stopping_rounds=10, verbose=False)
    print('Finished.')

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)
    # change the metric if you like
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    main()
