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
from pathlib import Path
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f'{ROOT_DIR}/data'
print(DATA_DIR)


def do_normalize(X, y):
    # import numpy as np
    # y = ss.fit_transform(np.array(y).reshape(len(y), 1))
    return StandardScaler().fit_transform(X), scale(y)


def do_split(X, y, ratio=0.2, seed=None):
    return train_test_split(X, y, test_size=ratio, random_state=seed)


class DataHelper:
    def __init__(self, name):
        self.data_dir = Path(DATA_DIR)
        self.name = name
        self.df, self.X, self.y = self._get_dataset()

    def _get_dataset(self):
        if self.name == 'boston':
            return self._boston()
        elif self.name == 'concrete':
            return self._concrete()
        elif self.name == 'biking':
            return self._biking()
        else:
            raise ValueError('{} is not yet implemented !'.format(self.name))

    def _boston(self):
        df = pd.read_csv(self.data_dir / 'boston/boston.csv')
        # we drop column 0 since it's only row index
        df.drop(df.columns[0], axis=1, inplace=True)
        target = 'medv'
        y = df[target]
        X = df.drop(columns=[target], axis=1)
        return df, X, y

    def _concrete(self):
        df = pd.read_csv(self.data_dir / 'concrete/concrete.csv')
        target = 'strength'
        y = df[target]
        X = df.drop(columns=[target], axis=1)
        return df, X, y

    def _biking(self):
        df = pd.read_csv(self.data_dir / 'bike-sharing/day.csv')
        target = 'cnt'
        y = df[target]
        X = df.drop(
            [
                target, 'instant', 'dteday', 'yr', 'casual', 'registered'
            ],
            axis=1
        )
        return df, X, y
