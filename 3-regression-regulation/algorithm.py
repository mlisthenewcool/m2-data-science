import numpy as np
from operator import add
from functools import reduce
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# custom
from helper.hyperparameter import EstimatorSelectionHelper
from helper.data import DataHelper, do_split


DATASETS = [
    {'label': 'Linear', 'value': 'linear'},
    {'label': 'Sine curve', 'value': 'sin'},
]

ALGORITHMS = [
    {'label': 'Linear regression', 'value': 'linear-regression'},
    {'label': 'Polynomial regression', 'value': 'polynomial-regression'},
    {'label': 'RBF Kernel Ridge', 'value': 'kernel-ridge-rbf'},
    {'label': 'Ridge', 'value': 'ridge'},
    {'label': 'Lasso', 'value': 'lasso'}
    # {'label': 'Linear Kernel Ridge', 'value': 'kernel_ridge_linear'}
]


def linear_data(size, variance):
    """
    Générons des données selon un modèle linéaire et un p'tit peu de bruit.

    :param size: Le nombre de points à générer
    :param variance: La variance du bruit gaussien
    :return:
    """
    X = np.random.random((size, 1))
    noise = np.random.normal(0, variance, (size, 1))
    b0, b1 = np.random.random(2)
    y = b0 + b1 * X + noise

    return X, y, b0, b1


def non_linear_data(size, variance):
    """
    Générons des données selon un modèle périodique non linéaire.

    :param size: Le nombre de points à générer
    :param variance: La variance du bruit gaussien
    :return:
    """
    # x = np.arange(size).reshape((size, 1))
    x = np.random.random((size, 1)) * size / 2

    def data_func(var):
        return np.sin(var/10) + (var/50) ** 2 + np.random.normal(0, variance)

    return x, data_func(x)


def linear_regression(x, y):
    # vector_ones = np.ones((x.shape[0], 1))
    # x = np.concatenate((vector_ones, x), axis=1)

    return np.linalg.inv(x.T @ x) @ x.T @ y


def get_regression_points(x, coeffs):
    return reduce(add, [coeffs[i][0] * (x ** i) for i in range(len(coeffs))])


def regression_biking(X_train, X_test, y_train, y_test):
    # allez on apprend !
    clf_ridge = KernelRidge(kernel='rbf')
    clf_ridge.fit(X_train, y_train)

    clf_lasso = Lasso()
    clf_lasso.fit(X_train, y_train)

    clf_linear = LinearRegression()
    clf_linear.fit(X_train, y_train)

    # show me the numbers !
    print('[ERREUR DE PREDICTION]')
    print('Classifier Ridge : {}'.format(clf_ridge.score(X_test, y_test)))
    print('Classifier Lasso : {}'.format(clf_lasso.score(X_test, y_test)))
    print('Classifier Linear : {}'.format(clf_linear.score(X_test, y_test)))

    # erreurs d'apprentissage
    print('[ERREUR D\'APPRENTISSAGE]')
    print('Classifier Ridge : {}'.format(clf_ridge.score(X_train, y_train)))
    print('Classifier Lasso : {}'.format(clf_lasso.score(X_train, y_train)))
    print('Classifier Linear : {}'.format(clf_linear.score(X_train, y_train)))

    print('[RUNNING GRID SEARCH]')
    # cross validation
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }

    params = {
        'Lasso': {
            'alpha': [1, 5, 10, 20, 30, 50, 70, 100, 1_000, 10_000]
        },
        'Ridge': {
            'alpha': [1, 5, 10, 20, 30, 50, 70, 100, 1_000, 10_000]
        }
    }

    grid = EstimatorSelectionHelper(models, params)
    grid.fit(X_train, y_train, n_jobs=2)
    print(grid.score_summary(sort_by='mean_score', num_rows_per_estimator=5))


def regression_fake_data(dataset, algorithm, data_size, data_noise):
    if dataset == 'linear':
        X, y, b0, b1 = linear_data(data_size, data_noise)
    elif dataset == 'sin':
        X, y = non_linear_data(data_size, data_noise)
    else:
        raise ValueError(f'Dataset {dataset} is not implemented yet')

    print(f'EXPLORING FAKE DATA {dataset} : {data_size} / {data_noise}')

    # helpers
    vector_ones = np.ones((data_size, 1))
    X_with_ones = np.concatenate((vector_ones, X), axis=1)
    X_range = np.linspace(X.min(), X .max(), len(X)).reshape(-1, 1)
    results = []

    # allez on claque les algos
    if 'lin_reg' in algorithm:
        results.append(
            {
                'name': 'lin_reg',
                'predictions': get_regression_points(X_range, linear_regression(X_with_ones, y)),
                'score': 'not yet calculated'
            })

    if 'poly_reg' in algorithm:
        X_with_ones = np.concatenate((vector_ones, X, X ** 2, X ** 3), axis=1)
        results.append(
            {
                'name': f'Polynomial ({3}) regression',
                'predictions': get_regression_points(
                    X_range,linear_regression(X_with_ones, y)),
                'score': 'not yet calculated'
            })

    if 'ridge' in algorithm:
        algo = Ridge()
        algo.fit(X, y)

        results.append(
            {
                'name': 'Ridge',
                'predictions': algo.predict(X_range),
                'score': algo.score(X, y)
            })

    if 'kernel_ridge_rbf' in algorithm:
        algo = KernelRidge(kernel='rbf')
        algo.fit(X, y)
        results.append(
            {
                'name': 'RBF_KernelRidge',
                'predictions': algo.predict(X_range),
                'score': algo.score(X, y)
            })

    for result in results:
        print(f"{result['name']} :: {result['score']}")
        print('=' * 40)


if __name__ == '__main__':
    # let's explore some fake data
    regression_fake_data('linear', [algo['value'] for algo in ALGORITHMS],
                         100, 0.05)
    regression_fake_data('sin', [algo['value'] for algo in ALGORITHMS],
                         100, 0.05)

    print('=' * 80)

    # let's explore some real data
    data = DataHelper('biking')
    X_train, X_test, y_train, y_test = do_split(
        data.X, data.y, ratio=0.2, seed=42)
    regression_biking(X_train, X_test, y_train, y_test)
