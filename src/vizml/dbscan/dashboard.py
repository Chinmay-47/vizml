import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from vizml._dashboard_configs import DASH_STYLE
from vizml.dbscan.clustering import DBScan


class DashBoard:
    """Class to run a dashboard for DBScan."""

    _dbscan_visualizer = dash.Dash(name="dbscan")

    _dbscan_visualizer.layout = html.Div([
        html.H1("Density Based Spatial Clustering of Applications with Noise",
                style=DASH_STYLE),
        dcc.Tabs(id='plot-tabs', value='tab-1', children=[
            dcc.Tab(label='Data Points', value='tab-1',
                    style=DASH_STYLE),
            dcc.Tab(label='Clusters', value='tab-2',
                    style=DASH_STYLE),
            dcc.Tab(label='Silhouette Plot', value='tab-3',
                    style=DASH_STYLE),
            dcc.Tab(label='Cluster Frequencies', value='tab-4',
                    style=DASH_STYLE),
            dcc.Tab(label='Model Metrics', value='tab-5',
                    style=DASH_STYLE),
        ], style=DASH_STYLE),
        html.Div([
            dcc.Slider(
                id="min-dist",
                min=0.1,
                max=2,
                step=0.1,
                marks={str(float(i/10)): "max_dist {}".format(i/10) for i in range(2, 21, 3)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=0.2,
                dots=False)
        ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
        html.Div([
            dcc.Slider(
                id="min-neighbours",
                min=1,
                max=30,
                step=1,
                marks={str(i): "min_neighbours {}".format(i) for i in range(2, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=5,
                dots=False)
        ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
        html.Div([
            dcc.Slider(
                id="no-points",
                min=50,
                max=300,
                step=1,
                marks={str(i): "{} points".format(i) for i in range(50, 310, 50)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=100,
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
                id='no-dimensions',
                options=[
                    {'label': '2D', 'value': '2d'},
                    {'label': '3D', 'value': '3d'},
                ],
                value='2d',
                style={'display': 'inline-block', 'width': '40%'}
            )
        ], style=DASH_STYLE),
        dcc.Graph('plot'),
        dcc.Store(id='random-state'),
        dcc.Store(id='plot1'),
        dcc.Store(id='plot2'),
        dcc.Store(id='plot3'),
        dcc.Store(id='plot4'),
        dcc.Store(id='plot5')
    ], style=DASH_STYLE)

    @staticmethod
    @_dbscan_visualizer.callback(
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
    @_dbscan_visualizer.callback(
        Output(component_id='plot1', component_property='figure'),
        Output(component_id='plot2', component_property='figure'),
        Output(component_id='plot3', component_property='figure'),
        Output(component_id='plot4', component_property='figure'),
        Output(component_id='plot5', component_property='figure'),
        Input(component_id='random-state', component_property='value'),
        Input(component_id='no-points', component_property='value'),
        Input(component_id='min-dist', component_property='value'),
        Input(component_id='min-neighbours', component_property='value'),
        Input(component_id='no-dimensions', component_property='value')
    )
    def _init_clustering(random_state, num_points, max_dist, min_neighbours, num_dim):
        """Initialize K Means Clustering and store all the plots."""

        is_3d = False if num_dim == '2d' else True

        clu = DBScan(no_points=num_points, max_dist=max_dist, min_no_points=min_neighbours,
                     random_state=random_state, is_3d=is_3d)
        clu.train()

        return (clu.show_data(return_fig=True), clu.show_clusters(return_fig=True),
                clu.show_silhouette_plot(return_fig=True), clu.show_freq_distribution(return_fig=True),
                clu.show_metrics(return_fig=True))

    @staticmethod
    @_dbscan_visualizer.callback(
        Output(component_id='plot', component_property='figure'),
        Input(component_id='plot-tabs', component_property='value'),
        Input(component_id='plot1', component_property='figure'),
        Input(component_id='plot2', component_property='figure'),
        Input(component_id='plot3', component_property='figure'),
        Input(component_id='plot4', component_property='figure'),
        Input(component_id='plot5', component_property='figure')
    )
    def _update_plots(plot_tab, fig1, fig2, fig3, fig4, fig5):
        """Updates the plot based on the chosen tab."""

        if plot_tab == "tab-1":
            return fig1
        elif plot_tab == "tab-2":
            return fig2
        elif plot_tab == "tab-3":
            return fig3
        elif plot_tab == "tab-4":
            return fig4
        else:
            return fig5

    def run(self):
        """Runs a dashboard on localhost to visualize DBScan."""

        self._dbscan_visualizer.run_server()


if __name__ == '__main__':
    DashBoard().run()
