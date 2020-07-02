import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from plotly import graph_objs as go

from helper.dash_components import *
from algorithm import *

app = dash.Dash(__name__)
server = app.server


def get_file_content_as_string(path):
    with open(path, 'r') as file:
        return file.read()


real_algorithms = [algo for algo in ALGORITHMS
                   if algo['value'] in ['ridge', 'lasso', 'linear-regression']]
gs_algorithms = [algo for algo in ALGORITHMS
                 if algo['value'] in ['ridge', 'lasso']]

fake_data_controls = [
    MyControl(
        dcc.Dropdown, 'dataset', id='fake-dataset',
        value="linear", options=DATASETS
    ),

    MyControl(
        dcc.Dropdown, 'algorithm', id='fake-algorithm',
        value=[algo['value'] for algo in ALGORITHMS],
        options=ALGORITHMS, multi=True
    ),

    MyControl(
        dcc.Input, 'number of points', id="fake-size",
        value=500, type="number"
    ),

    MyControl(
        dcc.Input, 'noise', id="fake-noise",
        value=0.05, type="number"
    ),
]

real_data_controls = [
    MyControl(
        dcc.Dropdown, 'algorithm', id='real-algorithm',
        value=[algo['value'] for algo in real_algorithms],
        options=real_algorithms, multi=True
    ),

    MyControl(
        dcc.Slider, 'train/test ratio', id='real-train-test-ratio',
        min=0.1, max=0.9, step=0.1, value=0.2
        # marks={i/10: str(i) for i in range(0, 10)}
    )
]

grid_search_controls = [
    MyControl(
        dcc.Dropdown, 'algorithm', id='grid-search-algorithm',
        value=[algo['value'] for algo in gs_algorithms],
        options=gs_algorithms, multi=True
    ),

    MyControl(
        dcc.Slider, 'train/test ratio', id='grid-search-train-test-ratio',
        min=0.1, max=0.9, step=0.1, value=0.2
        # marks={i/10: str(i) for i in range(0, 10)}
    )
]

my_title = "ds_2_regression"

app.layout = html.Main([
    # header
    html.Section(
        className="section",
        children=[
            html.Div(
                className="container",
                children=[
                    # title
                    html.H1(
                        className="is-size-1",
                        children="TP2 on regression"),
                    # description
                    html.P(
                        className="description",
                        children="Exploration of various regression methods. \
                        We will use linear, polynomial, Ridge and Lasso \
                        regressions."
                    ),
                ]
            )
        ]
    ),

    # fake data, section 1
    html.Section(
        className="section",
        children=[
            html.Div(
                className="container",
                children=[
                    # title
                    html.H3(
                        className="is-size-3",
                        children="Fake data"
                    ),

                    # user controls
                    MyControlBox(fake_data_controls),

                    # graph
                    MyGraphBox(
                        id='fake-graph',
                        figure={
                            'data': [],
                            'layout': DefaultGraphLayout()
                        }
                    )
                ],
            )
        ]
    ),

    # real data, section 2
    html.Section(
        className="section",
        children=[
            html.Div(
                className="container",
                children=[
                    # title
                    html.H3('Real data'),

                    # user controls
                    MyControlBox(real_data_controls),

                    # graph
                    MyGraphBox(
                        id='real-graph',
                        figure={
                            'data': [],
                            'layout': DefaultGraphLayout()
                        }
                    )
                ],
            )
        ]
    ),

    # grid search on real data, section 3
    html.Section(
        className="section",
        children=[
            html.Div(
                className="container",
                children=[
                    # title
                    html.H4('Grid search on real data'),

                    # user controls
                    MyControlBox(grid_search_controls),

                    # graph
                    MyGraphBox(
                        id='grid-search',
                        figure={
                            'data': [],
                            'layout': DefaultGraphLayout()
                        }
                    )
                ],
            )
        ]
    )
])


@app.callback(
    Output('fake-graph', 'figure'),
    [
        Input('fake-dataset', 'value'),
        Input('fake-algorithm', 'value'),
        Input('fake-size', 'value'),
        Input('fake-noise', 'value')
    ])
def update_fake_graph(dataset, algorithm, size, noise):
    if not dataset or not algorithm or not size:
        return {
            'data': [],
            'layout': DefaultGraphLayout()
        }

    if dataset == 'linear':
        X, y, b0, b1 = linear_data(size, noise)
    elif dataset == 'sin':
        X, y = non_linear_data(size, noise)
    else:
        raise ValueError(f'Dataset {dataset} is not implemented yet')

    # helpers
    vector_ones = np.ones((size, 1))
    X_with_ones = np.concatenate((vector_ones, X), axis=1)
    X_range = np.linspace(X.min(), X .max(), len(X)).reshape(-1, 1)
    results = []

    # allez on claque les algos
    if 'linear-regression' in algorithm:
        results.append(
            {
                'name': 'linear-regression',
                'predictions': get_regression_points(X_range, linear_regression(X_with_ones, y)),
                'score': 'not yet calculated'
            })

    if 'polynomial-regression' in algorithm:
        X_with_ones = np.concatenate((vector_ones, X, X ** 2, X ** 3), axis=1)
        results.append(
            {
                'name': f'Polynomial regression (degree : {3})',
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

    """
    if 'kernel_ridge_linear' in algorithm:
        algo = KernelRidge(kernel='linear')
        algo.fit(X_with_ones, y)
        results.append(
            {
                'name': 'Linear_KernelRidge',
                'predictions': algo.predict(X_with_ones),
                'score': algo.score(X_with_ones, y)
            })
    """

    if 'kernel-ridge-rbf' in algorithm:
        algo = KernelRidge(kernel='rbf')
        algo.fit(X, y)
        results.append(
            {
                'name': 'RBF_KernelRidge',
                'predictions': algo.predict(X_range),
                'score': algo.score(X, y)
            })

    traces = []
    for result in results:
        traces.append(
            go.Scatter(
                x=X_range.squeeze(),
                y=result['predictions'].squeeze(),
                mode='lines',
                name=f"{result['name']}"  # {result['score']}"
            )
        )

    traces.append(
        go.Scatter(
            x=X.squeeze(),
            y=y.squeeze(),
            mode='markers',
            name='Random data'
        ),
    )

    graph_layout = {
        'legend': {
            'orientation': 'h'
        }
    }

    return {
        'data': traces,
        'layout': {**DefaultGraphLayout(), **graph_layout}
    }


@app.callback(
    Output('real-graph', 'figure'),
    [
        Input('real-algorithm', 'value'),
        Input('real-train-test-ratio', 'value')
    ])
def update_real_graph(algorithm, train_test_ratio):
    # let's explore some real data

    data = DataHelper('biking')
    X_train, X_test, y_train, y_test = do_split(
        data.X, data.y, ratio=train_test_ratio, seed=42)

    results = []
    if 'ridge' in algorithm:
        algo = Ridge()
        algo.fit(X_train, y_train)
        results.append(
            {
                'name': 'Ridge',
                'train_score': algo.score(X_train, y_train),
                'test_score': algo.score(X_test, y_test)
            })

    if 'lasso' in algorithm:
        algo = Lasso()
        algo.fit(X_train, y_train)
        results.append(
            {
                'name': 'Lasso',
                'train_score': algo.score(X_train, y_train),
                'test_score': algo.score(X_test, y_test)
            })

    if 'linear-regression' in algorithm:
        algo = LinearRegression()
        algo.fit(X_train, y_train)
        results.append(
            {
                'name': 'LinearRegression',
                'train_score': algo.score(X_train, y_train),
                'test_score': algo.score(X_test, y_test)
            })

    import pandas as pd
    results = pd.DataFrame(results)

    traces = [
        go.Bar(
            x=results["name"],
            y=results["test_score"],
            name="test_score"
        ),
        go.Bar(
            x=results["name"],
            y=results["train_score"],
            name="train_score"
        )
    ]

    custom_graph_layout = {
        'legend': {
            'orientation': 'h'
        }
    }

    return {
        'data': traces,
        'layout': {**DefaultGraphLayout(), **custom_graph_layout}
    }


@app.callback(
    Output('grid-search', 'figure'),
    [
        Input('grid-search-algorithm', 'value'),
        Input('grid-search-train-test-ratio', 'value')
    ])
def update_grid_search(algorithm, train_test_ratio):
    # let's explore some real data

    data = DataHelper('biking')
    X_train, X_test, y_train, y_test = do_split(
        data.X, data.y, ratio=train_test_ratio, seed=42)

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

    gs = grid.score_summary(sort_by='mean_score',
                            num_rows_per_estimator=5)

    traces = [
        go.Table(
            header=dict(
                values=list(
                    gs.columns
                )
            ),
            cells=dict(
                values=[
                    gs.estimator,
                    gs.mean_score,
                    gs.std_score,
                    gs.min_score,
                    gs.max_score,
                    gs.alpha
                ]
            )
        )
    ]

    custom_graph_layout = {
        'font': {
            'color': 'darkblue'
        }
    }

    return {
        'data': traces,
        'layout': {**DefaultGraphLayout(), **custom_graph_layout}
    }


if __name__ == '__main__':
    app.run_server(debug=True)
