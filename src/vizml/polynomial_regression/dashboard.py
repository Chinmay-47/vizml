import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from vizml._dashboard_configs import DASH_STYLE
from vizml.polynomial_regression.regression import PolynomialRegression


class DashBoard:
    """Class to run a dashboard for Polynomial Regression."""

    _polynomial_regression_visualizer = dash.Dash(name="polynomial_regression")

    _polynomial_regression_visualizer.layout = html.Div([
        html.H1("Polynomial Regression",
                style=DASH_STYLE),
        dcc.Tabs(id='plot-tabs', value='tab-1', children=[
            dcc.Tab(label='Data Points', value='tab-1',
                    style=DASH_STYLE),
            dcc.Tab(label='Regression Curve', value='tab-2',
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
                marks={str(i): "{} points".format(i) for i in range(10, 100, 20)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=10,
                dots=False)
        ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
        html.Div([
            dcc.Slider(
                id="degree",
                min=1,
                max=10,
                step=1,
                marks={str(i): "degree {}".format(i) for i in range(1, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=1,
                dots=True)
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
        dcc.Store(id='random-state'),
        dcc.Store(id='plot1'),
        dcc.Store(id='plot2'),
        dcc.Store(id='plot3'),
    ], style=DASH_STYLE)

    @staticmethod
    @_polynomial_regression_visualizer.callback(
        Output(component_id='random-state', component_property='value'),
        Input(component_id='randomize', component_property='value')
    )
    def _update_random_state(random_val):
        """Sets a random state based on the input."""

        if random_val == 'random':
            np.random.seed(seed=None)
            random_state = np.random.randint(low=1, high=100000)
            return random_state

        return -1

    @staticmethod
    @_polynomial_regression_visualizer.callback(
        Output(component_id='plot1', component_property='figure'),
        Output(component_id='plot2', component_property='figure'),
        Output(component_id='plot3', component_property='figure'),
        Input(component_id='random-state', component_property='value'),
        Input(component_id='no-points', component_property='value'),
        Input(component_id='linearly-increasing', component_property='value'),
        Input(component_id='degree', component_property='value')
    )
    def _init_regressor(random_state, num_points, is_lin_inc, degree):
        """"Initialize Polynomial Regression and store all the plots."""

        is_increasing = True if is_lin_inc == 'increasing' else False

        reg1 = PolynomialRegression(no_points=num_points, random_state=random_state,
                                    is_increasing=is_increasing, degree=degree)
        reg1.train()

        return (reg1.show_data(return_fig=True), reg1.show_regression_curve(return_fig=True),
                reg1.show_error_scores(return_fig=True))

    @staticmethod
    @_polynomial_regression_visualizer.callback(
        Output(component_id='plot', component_property='figure'),
        Input(component_id='plot-tabs', component_property='value'),
        Input(component_id='plot1', component_property='figure'),
        Input(component_id='plot2', component_property='figure'),
        Input(component_id='plot3', component_property='figure')
    )
    def _update_plots(plot_tab, fig1, fig2, fig3):
        """Updates the plot based on the chosen tab."""

        if plot_tab == "tab-1":
            return fig1
        elif plot_tab == "tab-2":
            return fig2
        else:
            return fig3

    def run(self):
        """Runs a dashboard on localhost to visualize Polynomial Regression."""

        self._polynomial_regression_visualizer.run_server()


if __name__ == '__main__':
    DashBoard().run()
