import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from vizml.multi_linear_regression.regression import (OrdinaryLeastSquaresRegression, LassoRegression,
                                                      RidgeRegression, MultiLinearRegression)
from vizml._dashboard_configs import DASH_STYLE


multi_linear_regression_visualizer = dash.Dash(name="multi_linear_regression")

multi_linear_regression_visualizer.layout = html.Div([
    html.H1("Multi Linear Regression",
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
        dcc.Tab(label='Regression Plane', value='tab-2',
                style=DASH_STYLE),
        dcc.Tab(label='Error Metrics', value='tab-3',
                style=DASH_STYLE),
    ], style=DASH_STYLE),
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
    dcc.Graph('plot'),
    dcc.Slider(
        id="no-points",
        min=10,
        max=100,
        step=1,
        marks={str(i): "{} points".format(i) for i in range(10, 110, 10)},
        value=10,
        dots=False,
    ),
    dcc.Store(id='plot1'),
    dcc.Store(id='plot2'),
    dcc.Store(id='plot3')
], style=DASH_STYLE)


@multi_linear_regression_visualizer.callback(
    Output(component_id='plot1', component_property='figure'),
    Output(component_id='plot2', component_property='figure'),
    Output(component_id='plot3', component_property='figure'),
    Input(component_id='type-tabs', component_property='value'),
    Input(component_id='randomize', component_property='value'),
    Input(component_id='no-points', component_property='value'),
    Input(component_id='linearly-increasing', component_property='value')
)
def init_regressor(type_tab, val, no_points, is_inc):
    """Initializes the regressor and stores the initial plots."""

    randomize = True if val == 'random' else False
    is_increasing = True if is_inc == 'increasing' else False
    reg: MultiLinearRegression

    if type_tab == 'tab-1':
        reg = LassoRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    elif type_tab == 'tab-2':
        reg = RidgeRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    else:
        reg = OrdinaryLeastSquaresRegression(randomize=randomize, no_points=no_points, is_increasing=is_increasing)

    reg.train()

    return (reg.show_data(return_fig=True), reg.show_regression_plane(return_fig=True),
            reg.show_error_scores(return_fig=True))


@multi_linear_regression_visualizer.callback(
    Output(component_id='plot', component_property='figure'),
    Input(component_id='plot-tabs', component_property='value'),
    Input(component_id='plot1', component_property='figure'),
    Input(component_id='plot2', component_property='figure'),
    Input(component_id='plot3', component_property='figure')
)
def update_graph(plot_tab, plot1, plot2, plot3):
    """Switches plot based on selection"""

    if plot_tab == 'tab-1':
        return plot1
    elif plot_tab == 'tab-2':
        return plot2
    else:
        return plot3


if __name__ == '__main__':
    multi_linear_regression_visualizer.run_server()
