# customs
from webapp.common.helpers.hyperparameter import EstimatorSelectionHelper
from webapp.common.helpers.data import DataHelper, do_split

from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

g_datasets = ['concrete', 'biking']


def explore_coefficients(dataset, alphas):
    ###
    if dataset in g_datasets:
        data = DataHelper(dataset)
        X_train, X_test, y_train, y_test = do_split(
            data.X, data.y, ratio=0.2, seed=42
        )
    else:
        raise ValueError(f'{dataset} dataset is not available here.')

    ###
    coeffs = {
        'Ridge': [],
        'Lasso': [],
        'ElasticNet': []
    }
    scores = {
        'Ridge': [],
        'Lasso': [],
        'ElasticNet': []
    }

    for alpha in alphas:
        # RIDGE
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_train, y_train)
        coeffs['Ridge'].append(ridge.coef_)
        scores['Ridge'].append(
            (
                ridge.score(X_train, y_train),
                ridge.score(X_test, y_test)
            )
        )

        # LASSO
        lasso = Lasso(alpha=alpha, fit_intercept=False)
        lasso.fit(X_train, y_train)
        coeffs['Lasso'].append(lasso.coef_)
        scores['Lasso'].append(
            (
                lasso.score(X_train, y_train),
                lasso.score(X_test, y_test)
            )
        )

        elasticnet = ElasticNet(alpha=alpha, l1_ratio=0.5)
        elasticnet.fit(X_train, y_train)
        coeffs['ElasticNet'].append(elasticnet.coef_)
        scores['ElasticNet'].append(
            (
                elasticnet.score(X_train, y_train),
                elasticnet.score(X_test, y_test)
            )
        )

    return coeffs, scores


def launch_grid_search(dataset, alphas, l1_l2_ratios):
    if dataset in g_datasets:
        data = DataHelper(dataset)
        X, y = data.X, data.y
    else:
        raise ValueError(f'{dataset} dataset is not available here.')

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Elasticnet': ElasticNet()
    }

    params = {
        'Ridge': {
            'alpha': alphas
        },
        'Lasso': {
            'alpha': alphas,
            'random_state': [0],
            'max_iter': [100_000]
        },
        'Elasticnet': {
            'alpha': alphas,
            'l1_ratio': l1_l2_ratios
        }
    }

    grid = EstimatorSelectionHelper(models, params)
    grid.fit(X, y, n_jobs=2)
    return grid.score_summary(sort_by='mean_score',
                              num_rows_per_estimator=5)


if __name__ == '__main__':
    alphas = np.linspace(0, 50, 50)
    l1_l2_ratios = np.linspace(0.1, 1.0, 10)

    explore_coefficients('concrete', alphas)
    launch_grid_search('concrete', alphas, l1_l2_ratios)
