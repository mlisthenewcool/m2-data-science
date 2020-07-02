from matplotlib import pyplot as plt
import numpy as np

def td():
    points_x = [110, 125, 152, 172, 190, 208, 220, 242, 253, 270, 290]
    points_y = [187, 225, 305, 318, 367, 365, 400, 435, 450, 506, 558]

    a = 1.89
    b = -8.78

    f = lambda x: a * x + b
    x = np.array([min(points_x), max(points_x)])

    plt.scatter(points_x, points_y)
    plt.plot(x, f(x), lw=2.5, c="k", label="fit line between min and max")

    plt.legend()
    plt.show()


def plot_donnees(x, y):
    points = np.array([min(x), max(x)])
    plt.scatter(points, x)
    plt.plot(x, f(x), lw=2.5, c="k", label="fit line between min and max")

    plt.legend()
    plt.show()


def generer_donnees(nb_points=100):
    betas = np.random.random((2, 1))
    points_x = np.random.random((nb_points, 1))
    points_x = np.concatenate((points_x, np.ones((nb_points, 1))), axis=1)

    # le bruit & la fonction de génération
    # epsilons = np.random.rand((nb_points, 1))
    # fonction_generation = lambda var: coefficient_a * var + coefficient_b

    y = points_x @ betas + np.random.normal(0, 0.05, (nb_points, 1))

    # on enregistre les y
    # points_y = fonction_generation(points_x) + epsilons

    plt.scatter(points_x[:, 0], y)
    # plt.plot(x, f(x), lw=2.5, c="k", label="fit line between min and max")

    plt.legend()
    plt.show()

    return points_x, y


def generer_donnees_non_lineaires(nb_points=50):
    betas = np.random.random((2, 1))

    points_x = np.arange(nb_points).reshape((nb_points, 1))

    f = lambda var: np.sin(var / 10) + np.sin(var / 50) ** 2 + np.random.normal(0, 0.05)
    points_y = f(points_x)

    plt.scatter(points_x[:, 0], points_y)

    plt.legend()
    plt.show()

    return points_x, points_y


def regression_lineaire_moindres_carres(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


def regression_polynomiale(x, y):
    points_x = np.concatenate((np.ones(len(x), x, x ** 2, x ** 3)), axis=1)
    return points_x @ regression_lineaire_moindres_carres(points_x, y)


def generer_donnees_non_lineaires(nb_points=50):
    betas = np.random.random((2, 1))

    points_x = np.arange(nb_points).reshape((nb_points, 1))

    f = lambda var: np.sin(var / 10) + np.sin(var / 50) ** 2 + np.random.normal(0, 0.05)
    points_y = f(points_x)

    plt.scatter(points_x[:, 0], points_y)

    plt.legend()
    plt.show()

    return points_x, points_y


def regression_polynomiale(x, y):
    print(x.shape)
    print(x[0].shape)
    print(y.shape)
    print(y[0].shape)
    points_x = np.concatenate((np.ones(len(x)), x, x ** 2, x ** 3), axis=1)
    print(points_x.shape)
    return points_x @ regression_lineaire_moindres_carres(points_x, y)


if __name__ == '__main__':
    X, Y = generer_donnees_non_lineaires()
    # print(regression_lineaire_moindres_carres(X, Y))
    print(regression_polynomiale(X, Y))
