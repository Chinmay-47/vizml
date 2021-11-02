import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from vizml.simple_linear_regression.regression import (OrdinaryLeastSquaresRegression, LassoRegression,
                                                       RidgeRegression, SimpleLinearRegression)
from vizml._dashboard_configs import DASH_STYLE


simple_linear_regression_visualizer = dash.Dash(name="simple_linear_regression")

simple_linear_regression_visualizer.layout = html.Div([
    html.H1("Simple Linear Regression",
            style=DASH_STYLE),
    html.Div([
        dcc.Dropdown(
            id='linear-reg-choice',
            options=[{'label': i, 'value': i} for i in
                     ['OrdinaryLeastSquaresRegression', 'LassoRegression', 'RidgeRegression']],
            value='OrdinaryLeastSquaresRegression',
            style=DASH_STYLE | {'width': '100%'})
    ]),
    html.Div([
        dcc.RadioItems(
            id='randomize',
            options=[
                {'label': 'Initial', 'value': 'initial'},
                {'label': 'Random', 'value': 'random'},
            ],
            value='initial',
        ),
        dcc.RadioItems(
            id='linearly-increasing',
            options=[
                {'label': 'Increasing', 'value': 'increasing'},
                {'label': 'Decreasing', 'value': 'decreasing'},
            ],
            value='increasing',
        )
    ], style=DASH_STYLE | {'width': '48%', 'display': 'inline-block'}),
    dcc.Slider(
        id="no-points",
        min=10,
        max=100,
        step=1,
        marks={str(i): str(i) for i in range(10, 105, 5)},
        value=10,
        dots=False,
    ),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Data Points', value='tab-1',
                style=DASH_STYLE),
        dcc.Tab(label='Regression Line', value='tab-2',
                style=DASH_STYLE),
        dcc.Tab(label='Error Metrics', value='tab-3',
                style=DASH_STYLE),
    ], style=DASH_STYLE),
    dcc.Graph('plot')
], style=DASH_STYLE)


@simple_linear_regression_visualizer.callback(
    Output(component_id='plot', component_property='figure'),
    Input(component_id='linear-reg-choice', component_property='value'),
    Input(component_id='randomize', component_property='value'),
    Input(component_id='no-points', component_property='value'),
    Input(component_id='tabs-example', component_property='value'),
    Input(component_id='linearly-increasing', component_property='value')
)
def update_graph(option, val, no_points, tab, is_inc):

    randomize = True if val == 'random' else False
    is_increasing = True if is_inc == 'increasing' else False
    reg: SimpleLinearRegression

    if option == 'LassoRegression':
        reg = LassoRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    elif option == 'RidgeRegression':
        reg = RidgeRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    else:
        reg = OrdinaryLeastSquaresRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    if tab == 'tab-1':
        return reg.show_data(return_fig=True)
    elif tab == 'tab-2':
        reg.train()
        return reg.show_regression_line(return_fig=True)
    else:
        reg.train()
        return reg.show_error_scores(return_fig=True)


if __name__ == '__main__':
    simple_linear_regression_visualizer.run_server()
