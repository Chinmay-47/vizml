import itertools

import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from vizml._dashboard_configs import DASH_STYLE
from vizml.simple_linear_regression.regression import (OrdinaryLeastSquaresRegression, LassoRegression,
                                                       RidgeRegression)


simple_linear_regression_visualizer = dash.Dash(name="simple_linear_regression")

simple_linear_regression_visualizer.layout = html.Div([
    html.H1("Simple Linear Regression",
            style=DASH_STYLE),
    html.Div([
        dcc.Tabs(id='type-tabs', value='tab-1', children=[
            dcc.Tab(label='Ordinary Least Squares Regression', value='tab-1',
                    style=DASH_STYLE),
            dcc.Tab(label='Lasso Regression', value='tab-2',
                    style=DASH_STYLE),
            dcc.Tab(label='Ridge Regression', value='tab-3',
                    style=DASH_STYLE),
        ], style=DASH_STYLE)
    ]),
    dcc.Tabs(id='plot-tabs', value='tab-1', children=[
        dcc.Tab(label='Data Points', value='tab-1',
                style=DASH_STYLE),
        dcc.Tab(label='Regression Line', value='tab-2',
                style=DASH_STYLE),
        dcc.Tab(label='Error Metrics', value='tab-3',
                style=DASH_STYLE),
    ], style=DASH_STYLE),
    html.Div([
        dcc.Slider(
            id="no-points",
            min=10,
            max=100,
            step=1,
            marks={str(i): str(i) for i in range(10, 110, 10)},
            tooltip={"placement": "bottom", "always_visible": False},
            value=10,
            dots=False)
    ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
    html.Div([
        dcc.RadioItems(
            id='randomize',
            options=[
                {'label': 'Initial', 'value': 'initial'},
                {'label': 'Random', 'value': 'random'},
            ],
            value='initial',
            style={'display': 'inline-block', 'width': '40%'}
        ),
        dcc.RadioItems(
            id='linearly-increasing',
            options=[
                {'label': 'Increasing', 'value': 'increasing'},
                {'label': 'Decreasing', 'value': 'decreasing'},
            ],
            value='increasing',
            style={'display': 'inline-block', 'width': '40%'}
        )
    ], style=DASH_STYLE),
    dcc.Graph('plot'),
    dcc.Store(id='plot1'),
    dcc.Store(id='plot2'),
    dcc.Store(id='plot3'),
    dcc.Store(id='plot4'),
    dcc.Store(id='plot5'),
    dcc.Store(id='plot6'),
    dcc.Store(id='plot7'),
    dcc.Store(id='plot8'),
    dcc.Store(id='plot9')
], style=DASH_STYLE)


@simple_linear_regression_visualizer.callback(
    Output(component_id='plot1', component_property='figure'),
    Output(component_id='plot2', component_property='figure'),
    Output(component_id='plot3', component_property='figure'),
    Output(component_id='plot4', component_property='figure'),
    Output(component_id='plot5', component_property='figure'),
    Output(component_id='plot6', component_property='figure'),
    Output(component_id='plot7', component_property='figure'),
    Output(component_id='plot8', component_property='figure'),
    Output(component_id='plot9', component_property='figure'),
    Input(component_id='randomize', component_property='value'),
    Input(component_id='no-points', component_property='value'),
    Input(component_id='linearly-increasing', component_property='value')
)
def init_regressors(is_random, no_points, is_inc):
    """Initializes the regressor and stores the initial plots."""

    randomize = True if is_random == 'random' else False
    is_increasing = True if is_inc == 'increasing' else False

    if not randomize:
        reg1 = LassoRegression(no_points=no_points, is_increasing=is_increasing)
        reg2 = RidgeRegression(no_points=no_points, is_increasing=is_increasing)
        reg3 = OrdinaryLeastSquaresRegression(no_points=no_points, is_increasing=is_increasing)

    else:
        np.random.seed(seed=None)
        random_state = np.random.randint(low=1, high=100000)
        reg1 = LassoRegression(no_points=no_points, is_increasing=is_increasing, random_state=random_state)
        reg2 = RidgeRegression(no_points=no_points, is_increasing=is_increasing, random_state=random_state)
        reg3 = OrdinaryLeastSquaresRegression(no_points=no_points, is_increasing=is_increasing,
                                              random_state=random_state)

    reg1.train()
    reg2.train()
    reg3.train()

    return (reg1.show_data(return_fig=True), reg1.show_regression_line(return_fig=True),
            reg1.show_error_scores(return_fig=True), reg2.show_data(return_fig=True),
            reg2.show_regression_line(return_fig=True), reg2.show_error_scores(return_fig=True),
            reg3.show_data(return_fig=True), reg3.show_regression_line(return_fig=True),
            reg3.show_error_scores(return_fig=True))


@simple_linear_regression_visualizer.callback(
    Output(component_id='plot', component_property='figure'),
    Input(component_id='plot-tabs', component_property='value'),
    Input(component_id='type-tabs', component_property='value'),
    Input(component_id='plot1', component_property='figure'),
    Input(component_id='plot2', component_property='figure'),
    Input(component_id='plot3', component_property='figure'),
    Input(component_id='plot4', component_property='figure'),
    Input(component_id='plot5', component_property='figure'),
    Input(component_id='plot6', component_property='figure'),
    Input(component_id='plot7', component_property='figure'),
    Input(component_id='plot8', component_property='figure'),
    Input(component_id='plot9', component_property='figure')
)
def update_graph(plot_tab, type_tab, plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9):
    """Switches plot based on selection"""

    all_plots = [plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9]

    tab_types = ['tab-1', 'tab-2', 'tab-3']
    combs = list(itertools.product(*[tab_types, tab_types]))
    for i, comb in enumerate(combs):
        if (type_tab, plot_tab) == comb:
            return all_plots[i]

    return plot1


if __name__ == '__main__':
    simple_linear_regression_visualizer.run_server()
