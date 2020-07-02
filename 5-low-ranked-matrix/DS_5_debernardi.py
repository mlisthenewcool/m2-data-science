import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_low_rank_matrix

# run2
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso, Lasso, Ridge


def low_ranked_regression(data, target, rank):
    weights = np.linalg.inv(data.T @ data) @ data.T @ target

    # svd decomposition
    _, _, v = np.linalg.svd(data @ weights)
    best_v = v[:rank].T @ v[:rank]

    return weights @ best_v


def least_square_regression(data, target):
    return np.linalg.inv(data.T @ data) @ data.T @ target


def run_1():
    n_features, n_samples, rank = 10, 10, 4
    # low_ranked = make_low_rank_matrix(n_features, n_samples, rank)

    weights = np.random.random((n_features, rank))
    low_ranked = weights @ weights.T  # 100, 100

    # random data with noise
    data = np.random.random((n_samples, n_features))
    noise = np.random.normal(0, 0.05, (n_samples, n_features))
    target = data @ low_ranked + noise

    """       
    print(low_ranked.shape)
    print(data.shape)
    print(noise.shape)
    print(target.shape)
    """

    ranks = []
    eckart_res = []
    ls_res = []

    search_ranks = range(1, 30)

    for search_rank in search_ranks:
        #plt.imshow(low_ranked)
        #plt.show()

        # low ranked regression
        low_ranked_weights = low_ranked_regression(data, target, search_rank)
        #plt.imshow(low_ranked_weights)
        #plt.show()

        # least square regression
        least_square_weights = least_square_regression(data, target)
        #plt.imshow(least_square_weights)
        #plt.show()

        ranks.append(search_rank)
        eckart_res.append(np.linalg.norm(low_ranked - low_ranked_weights))
        ls_res.append(np.linalg.norm(low_ranked - least_square_weights))

    fig = go.Figure(
        data=[
            go.Scatter(
                y=eckart_res,
                mode='lines',
                name='Eckart-Young results'
            ),
            go.Scatter(
                y=ls_res,
                mode='lines',
                name='Least-Square results'
            ),
        ]
    )

    #from plotly.offline import iplot
    #iplot(fig)
    fig.show()

    # plot matrices
    results_lr = low_ranked_regression(data, target, rank)
    results_ols = least_square_regression(data, target)

    print('Ordinary Least Square')
    plt.imshow(results_ols)
    plt.show()

    print('Low Rank')
    plt.imshow(results_lr)
    plt.show()

    print('Original weights')
    plt.imshow(low_ranked)
    plt.show()


def run_2(param_input):
    rng = np.random.RandomState(42)

    # Generate some 2D coefficients with sine waves with random frequency and phase
    n_samples, n_features, n_tasks = 100, 30, 40
    n_relevant_features = 5
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    for k in range(n_relevant_features):
        coef[:, k] = np.sin((1. + rng.randn(1, 1)) * times + 3 * rng.randn(1, 1))

    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)

    print(X.shape)
    print(Y.shape)

    coef_ridge_ = np.array([Ridge(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

    coef_low_ranked_ = low_ranked_regression(X, Y, n_relevant_features)

    # #############################################################################
    # Plot support and time series
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.spy(coef_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'Lasso')
    plt.subplot(1, 2, 2)
    plt.spy(coef_multi_task_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'MultiTaskLasso')
    fig.suptitle('Coefficient non-zero location')

    feature_to_plot = param_input
    plt.figure()
    lw = 2
    plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
             label='Ground truth')
    plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue',
             linewidth=lw,
             label='Lasso')
    plt.plot(coef_ridge_[:, feature_to_plot], color='red',
             linewidth=lw,
             label='Ridge')
    plt.plot(coef_low_ranked_[:, feature_to_plot], color='magenta',
             linewidth=lw,
             label='LowRanked')
    plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold',
             linewidth=lw,
             label='MultiTaskLasso')
    plt.legend(loc='upper center')
    plt.axis('tight')
    plt.ylim([-1.1, 1.1])
    plt.show()


def run_3():
    import os
    import pandas as pd
    from functools import reduce

    data_path = '/home/hippo/Websites/ml-is-the-new-cool/data'

    df1 = pd.read_csv(f'{data_path}/restaurants/chefmozaccepts.csv')
    df2 = pd.read_csv(f'{data_path}/restaurants/chefmozcuisine.csv')
    df3 = pd.read_csv(f'{data_path}/restaurants/chefmozhours4.csv')
    df4 = pd.read_csv(f'{data_path}/restaurants/chefmozparking.csv')

    dfs = [df1, df2, df3, df4]

    df_restaurants = reduce(lambda left, right: pd.merge(left, right,
                                                         on='placeID'), dfs)

    print(df_restaurants.describe())


def run_4(feature_to_plot):
    # Generate some 2D coefficients with sine waves with random frequency and phase
    from sklearn.datasets import load_linnerud
    linnerud = load_linnerud()
    """
    print(linnerud.feature_names)
    print(linnerud.data)

    print(linnerud.target_names)
    print(linnerud.target)
    """
    X, Y = linnerud.data, linnerud.target

    # [print(y) for y in Y.T]

    coef_ridge_ = np.array([Ridge(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

    coef_low_ranked_ = low_ranked_regression(X, Y, 3)

    # #############################################################################
    # Plot support and time series
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.spy(coef_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'Lasso')
    plt.subplot(1, 2, 2)
    plt.spy(coef_multi_task_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'MultiTaskLasso')
    fig.suptitle('Coefficient non-zero location')

    plt.tight_layout()
    plt.figure()
    lw = 1
    """
    plt.plot(Y[:, feature_to_plot], color='seagreen', linewidth=lw,
             label='Ground truth')
    """
    plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue',
             linewidth=lw,
             label='Lasso')
    plt.plot(coef_ridge_[:, feature_to_plot], color='red',
             linewidth=lw,
             label='Ridge')
    plt.plot(coef_low_ranked_[:, feature_to_plot], color='magenta',
             linewidth=lw,
             label='LowRanked')
    plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold',
             linewidth=lw,
             label='MultiTaskLasso')
    plt.legend(loc='upper center')
    plt.axis('tight')
    plt.ylim([-1.1, 1.1])
    plt.tight_layout()
    plt.show()


def run_5(n_samples=100):
    from libtlda.iw import ImportanceWeightedClassifier
    clf = ImportanceWeightedClassifier(loss='quadratic', iwe='kmm')

    # X ~ N(0.5, 0.5²)
    # Z ~ N(0.0, 0.3²)

    x = np.random.normal(0.5, 0.5 ** 2, (n_samples, 1))
    z = np.random.normal(0, 0.3 ** 2, (n_samples, 1))

    x_noise = np.random.normal(0, 0.07, (n_samples, 1))
    z_noise = np.random.normal(0, 0.03, (n_samples, 1))

    def data_func(var):
        return var ** 3 - var

    y = data_func(x)
    y = np.array(y)
    y = y.ravel()

    # + bruit
    X = x + x_noise
    Z = z + z_noise

    # distribution différente à approximer avec une distrib initiale
    y_bis = data_func(z)
    y_bis = np.array(y_bis)
    y_bis = y_bis.ravel()

    print(X.shape)
    print(y.shape)
    print(Z.shape)
    print(y_bis.shape)

    clf.fit(X, y, Z)
    preds = clf.predict(Z)
    print(np.linalg.norm(preds - Z))

    from sklearn.linear_model import LinearRegression
    clf_linear = LinearRegression()
    clf_linear.fit(Z, y_bis)
    true_coefs = clf_linear.coef_

    # print(clf.get_weights())

    print(preds)

    # plot facilities
    x_range = np.linspace(-0.4, 1.2, 100)
    kmm_line = x_range * preds
    true_line = x_range * true_coefs


    plt.axis([-0.4, 1.2, -0.5, 1])
    plt.scatter(X, y, label='X points', color='blue', marker='o')
    plt.plot(x_range, data_func(x_range), label='X distribution', color='blue')

    plt.scatter(Z, y_bis, label='Z points', color='red', marker='+')
    plt.plot(x_range, kmm_line, label='Z kmm regression line', color='red')

    plt.plot(x_range, true_line, label='Z OLS line', color='black')
    plt.legend()
    plt.show()

    """
    fig = go.Figure(
        data=[
            go.Scatter(
                y=X,
                mode='markers',
                name='X markers'
            ),
            go.Scatter(
                x=Z,
                mode='markers',
                name='Z markers'
            ),
        ]
    )
    fig.show()
    """



def run_6():
    X = np.random.randn(10, 2)
    y = np.vstack((-np.ones((5,)), np.ones((5,))))
    Z = np.random.randn(10, 2)

    from libtlda.iw import ImportanceWeightedClassifier
    clf = ImportanceWeightedClassifier(loss='quadratic', iwe='kmm')

    clf.fit(X, y, Z)
    u_pred = clf.predict(Z)

    print(u_pred)


if __name__ == '__main__':
    #run_1()

    # run_2(0)

    # [run_2(x) for x in range(0, 5)]
    # [run_4(x) for x in range(0, 3)]

    run_5(100)
    # run_6()
