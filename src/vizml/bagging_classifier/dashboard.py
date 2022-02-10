import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from vizml._dashboard_configs import DASH_STYLE
from vizml.bagging_classifier.classification import BaggingClassifier


class DashBoard:
    """Class to run a dashboard for Bagging Classifier."""

    _bagging_classifier_visualizer = dash.Dash(name="bagging_classifier")

    _bagging_classifier_visualizer.layout = html.Div([
        html.H1("Bagging Classifier",
                style=DASH_STYLE),
        dcc.Tabs(id='plot-tabs', value='tab-1', children=[
            dcc.Tab(label='Data Points', value='tab-1',
                    style=DASH_STYLE),
            dcc.Tab(label='Decision Boundary', value='tab-2',
                    style=DASH_STYLE),
            dcc.Tab(label='Decision Probabilities', value='tab-3',
                    style=DASH_STYLE),
            dcc.Tab(label='Confusion Matrix', value='tab-4',
                    style=DASH_STYLE),
            dcc.Tab(label='Classification Metrics', value='tab-5',
                    style=DASH_STYLE),
        ], style=DASH_STYLE),
        html.Div([
            dcc.Slider(
                id="no-points",
                min=10,
                max=200,
                step=1,
                marks={str(i): str(i) for i in range(10, 210, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=50,
                dots=False)
        ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
        html.Div([
            dcc.Slider(
                id="n-estimators",
                min=1,
                max=50,
                step=1,
                marks={str(i): "{} estimators".format(i) for i in range(5, 49, 5)},
                tooltip={"placement": "bottom", "always_visible": False},
                value=10,
                dots=False)
        ], style={**DASH_STYLE, **{'margin-top': '5px', 'margin-bottom': '5px'}}),
        html.Div([
            dcc.RadioItems(
                id='base-classifier',
                options=[
                    {'label': 'Decision Tree base model', 'value': 'dt'},
                    {'label': 'Logistic Regression base model', 'value': 'lr'},
                    {'label': 'K Nearest Neighbours base model', 'value': 'knn'},
                    {'label': 'Naive Bayes base model', 'value': 'nb'},
                    {'label': 'Support Vector Machine base model', 'value': 'svm'}
                ],
                value='dt',
                style={'display': 'inline-block'}
            ),
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
                id='data-shape',
                options=[
                    {'label': 'linearly separable', 'value': 'linearly_separable'},
                    {'label': 'moon shaped', 'value': 'moon'},
                    {'label': 'circle shaped', 'value': 'circle'}
                ],
                value='linearly_separable',
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
            ),
        ], style=DASH_STYLE),
        dcc.Graph('plot'),
        dcc.Store(id='random-state'),
        dcc.Store(id='plot1'),
        dcc.Store(id='plot2'),
        dcc.Store(id='plot3'),
        dcc.Store(id='plot4'),
        dcc.Store(id='plot5'),
    ], style=DASH_STYLE)

    @staticmethod
    @_bagging_classifier_visualizer.callback(
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
    @_bagging_classifier_visualizer.callback(
        Output(component_id='plot1', component_property='figure'),
        Output(component_id='plot2', component_property='figure'),
        Output(component_id='plot3', component_property='figure'),
        Output(component_id='plot4', component_property='figure'),
        Output(component_id='plot5', component_property='figure'),
        Input(component_id='random-state', component_property='value'),
        Input(component_id='no-points', component_property='value'),
        Input(component_id='data-shape', component_property='value'),
        Input(component_id='no-dimensions', component_property='value'),
        Input(component_id='n-estimators', component_property='value'),
        Input(component_id='base-classifier', component_property='value')
    )
    def _init_classifier(random_state, no_points, data_shape, num_dim, n_estimators, base_classifier):
        """Initializes the classifier and stores the initial plots."""

        is_3d = False if num_dim == '2d' else True

        clf = BaggingClassifier(no_points=no_points, random_state=random_state, data_shape=data_shape,
                                is_3d=is_3d, n_estimators=n_estimators, base_classifier=base_classifier)

        clf.train()

        return (clf.show_data(return_fig=True), clf.show_decision_boundary(return_fig=True),
                clf.show_decision_probabilities(return_fig=True), clf.show_confusion_matrix(return_fig=True),
                clf.show_metrics(return_fig=True))

    @staticmethod
    @_bagging_classifier_visualizer.callback(
        Output(component_id='plot', component_property='figure'),
        Input(component_id='plot-tabs', component_property='value'),
        Input(component_id='plot1', component_property='figure'),
        Input(component_id='plot2', component_property='figure'),
        Input(component_id='plot3', component_property='figure'),
        Input(component_id='plot4', component_property='figure'),
        Input(component_id='plot5', component_property='figure'),
    )
    def _update_graph(plot_tab, plot1, plot2, plot3, plot4, plot5):
        """Switches plot based on selection."""

        plots = {'tab-1': plot1, 'tab-2': plot2, 'tab-3': plot3, 'tab-4': plot4, 'tab-5': plot5}

        return plots[plot_tab]

    def run(self):
        """Runs a dashboard on localhost to visualize Bagging Classifier."""

        self._bagging_classifier_visualizer.run_server()


if __name__ == '__main__':
    DashBoard().run()
