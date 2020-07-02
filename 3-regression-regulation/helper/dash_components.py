import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go


DEFAULT_CLASSNAME_USER_CONTROL = "user-control-box"
DEFAULT_CLASSNAME_GRAPH = "graph-box"


"""
# user controls
MyGraphBoxWithControls(
    fake_data_controls,
    title="Playing around with fake data",
    id='fake-graph',
    figure={
        'data': [],
        'layout': DefaultGraphLayout()
    }
)
"""

"""
marks={i: str(i) for i in range(100, 1_001, 100)}
"""



def MyControl(control, label=None, **kwargs):
    return html.Div(
        className="field is-horizontal",
        children=[
            # label
            html.Div(
                className="field-label is-small",
                children=html.Label(
                    className="label",
                    children=label
                )
            ) if label else None,

            # input
            html.Div(
                className="field-body",
                children=[
                    html.Div(
                        className="field",
                        children=[
                            html.Div(
                                className="control",
                                children=[
                                    control(
                                        className="input is-small",
                                        **kwargs
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def MyControlBox(controls):
    return html.Div(
        className="box",
        children=[
            html.Div(
                className="columns is-multiline is-vcentered",
                children=[
                    html.Div(
                        className="column is-6",
                        children=control
                    ) for control in controls
                ]
            )
        ]
    )


def MyGraphBox(title=None, **kwargs):
    return html.Div(
        className='box',
        children=[
            html.P(title) if title else None,
            dcc.Graph(**kwargs)
        ]
    )


def MyGraphBoxWithControls(controls, title=None, **kwargs):
    return html.Div(
        className='box',
        children=[
            html.P(className='description', children=title),

            html.Div(
                className="columns is-multiline is-vcentered",
                children=[
                    html.Div(
                        className="column is-6",
                        children=control
                    ) for control in controls
                ]
            ),
            dcc.Graph(**kwargs)
        ]
    )


def DefaultGraphLayout():
    font_color = 'rgb(251, 162, 26)'
    transparent_bg = 'rgba(0, 0, 0, 0)'
    # background_bg = 'rbga(31, 38, 48, 20)'

    return {
        'paper_bgcolor': transparent_bg,
        'plot_bgcolor': transparent_bg,
        'font': {
            'color': font_color
        }
    }


def DefaultFigureLayout(min_x, max_x, min_y, max_y):
    return go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        #margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        #showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        # dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[-0.5 + min_x, max_x + 0.5],
            showgrid=False,
            # nticks=rows,
            # fixedrange=True,
        ),
        yaxis=dict(
            range=[-0.5 + min_y, max_y + 0.5],
            showgrid=False,
            # showticklabels=False,
            # fixedrange=True,
            # rangemode="nonnegative",
            # zeroline=False,
        )
    )
