import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from vizml._dashboard_configs import DASH_STYLE
from vizml.k_means_clustering.clustering import KMeansClustering


k_means_clustering_visualizer = dash.Dash(name="k_means_clustering")

k_means_clustering_visualizer.layout = html.Div([
    html.H1("K Means Clustering",
            style=DASH_STYLE),
    dcc.Tabs(id='plot-tabs', value='tab-1', children=[
        dcc.Tab(label='Data Points', value='tab-1',
                style=DASH_STYLE),
        dcc.Tab(label='Clusters', value='tab-2',
                style=DASH_STYLE),
        dcc.Tab(label='Elbow Method Plot', value='tab-3',
                style=DASH_STYLE),
    ], style=DASH_STYLE),
    html.Div([
        dcc.Slider(
            id="no-clusters",
            min=1,
            max=20,
            step=1,
            marks={str(i): "{} clusters".format(i) for i in range(2, 21, 4)},
            tooltip={"placement": "bottom", "always_visible": False},
            value=3,
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
    dcc.Store(id='plot3')
], style=DASH_STYLE)


@k_means_clustering_visualizer.callback(
    Output(component_id='random-state', component_property='value'),
    Input(component_id='randomize', component_property='value')
)
def update_random_state(random_val):
    """Sets a random state based on the input."""

    if random_val == 'random':
        np.random.seed(seed=None)
        random_state = np.random.randint(low=1, high=100000)
        return random_state

    return -1


@k_means_clustering_visualizer.callback(
    Output(component_id='plot1', component_property='figure'),
    Output(component_id='plot2', component_property='figure'),
    Output(component_id='plot3', component_property='figure'),
    Input(component_id='random-state', component_property='value'),
    Input(component_id='no-points', component_property='value'),
    Input(component_id='no-clusters', component_property='value'),
    Input(component_id='no-dimensions', component_property='value')
)
def init_clustering(random_state, num_points, num_clusters, num_dim):
    """"Initialize K Means Clustering and store all the plots."""

    is_3d = False if num_dim == '2d' else True

    clu = KMeansClustering(no_points=num_points, no_clusters=num_clusters, random_state=random_state,
                           is_3d=is_3d)
    clu.train()

    return clu.show_data(return_fig=True), clu.show_clusters(return_fig=True), clu.show_elbow_plot(return_fig=True)


@k_means_clustering_visualizer.callback(
    Output(component_id='plot', component_property='figure'),
    Input(component_id='plot-tabs', component_property='value'),
    Input(component_id='plot1', component_property='figure'),
    Input(component_id='plot2', component_property='figure'),
    Input(component_id='plot3', component_property='figure')
)
def update_plots(plot_tab, fig1, fig2, fig3):
    """Updates the plot based on the chosen tab."""

    if plot_tab == "tab-1":
        return fig1
    elif plot_tab == "tab-2":
        return fig2
    else:
        return fig3


if __name__ == '__main__':
    k_means_clustering_visualizer.run_server()
