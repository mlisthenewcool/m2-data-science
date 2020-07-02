# -*- coding: utf-8 -*-
# Copyright 2019 Hippolyte L. DEBERNARDI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError('Some estimators are missing parameters: %s'
                             % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score',
                      num_rows_per_estimator=None):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'mean_score': round(np.mean(scores), 5),
                'std_score': round(np.std(scores), 5),
                'min_score': round(min(scores), 5),
                'max_score': round(max(scores), 5),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'mean_score', 'std_score', 'min_score',
                   'max_score']
        columns = columns + [c for c in df.columns if c not in columns]

        if num_rows_per_estimator:
            df = df.groupby('estimator')
            # df.apply(lambda _df: _df.sort_values(by=['mean_score']))
            df = df.head(num_rows_per_estimator)

        return df[columns]


if __name__ == '__main__':
    models = {
        'Perceptron': Perceptron(),
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC()
    }

    params = {
        'Perceptron': {
            'penalty': [None, 'l2', 'l1', 'elasticnet'],
            'alpha': np.linspace(0, 1, 50)
        },
        'LogisticRegression': {
            'penalty': ['l1', 'l2'],
        },
        'KNeighborsClassifier': {
            'n_neighbors': np.linspace(2, 15, 14),
            'weights': ['uniform', 'distance'],
            # 'algorithm': ['']
            'p': [1, 2]
        },
        'SVC': {
            'C': [1.0, 2.0, 3.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': np.linspace(2, 10, 9),
            # 'gamma' :
        }
    }

    datasets = {
        'Iris': load_iris(return_X_y=True),
        'Digits': load_digits(return_X_y=True),
        'BreastCancer': load_breast_cancer(return_X_y=True)
    }


    # iterate over datasets
    for ds_name, ds in datasets.items():
        print(ds_name)

        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42)

        #clf.fit(X_train, y_train)
        #score = clf.score(X_test, y_test)

        grid = EstimatorSelectionHelper(models, params)
        grid.fit(X_train, y_train, n_jobs=2)
        print(grid.score_summary(sort_by='mean_score', num_rows_per_estimator=5))
