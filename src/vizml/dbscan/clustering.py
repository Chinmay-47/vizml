import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.cluster import DBSCAN
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import Normal2DGenerator, Normal3DGenerator, NormalDataGenerator
from vizml.metrics.clustering_metrics import AllSilhouetteScores


class DBScan:
    """Class to perform and visualize Density Based Spatial Clustering of Applications with Noise."""

    def __init__(self, no_points: int = 100, min_no_points: int = 10, min_dist: float = 0.5,
                 randomize: bool = False, random_state: int = -1, is_3d: bool = False):

        self.min_no_points = min_no_points
        self.min_dist = min_dist
        self.clustering = DBSCAN(eps=self.min_dist, min_samples=self.min_no_points, n_jobs=-1)
        self.randomize = randomize
        self.is_3d = is_3d
        self.no_points = no_points
        dpgen: NormalDataGenerator
        if self.is_3d:
            dpgen = Normal3DGenerator(random=randomize, random_state=random_state)
            self.data_points = dpgen.generate(no_of_points=self.no_points)
            self.x1_values = self.data_points[:, 0]
            self.x2_values = self.data_points[:, 1]
            self.y_values = self.data_points[:, 2]
        else:
            dpgen = Normal2DGenerator(random=randomize, random_state=random_state)
            self.data_points = dpgen.generate(no_of_points=self.no_points)
            self.x_values = self.data_points[:, 0]
            self.y_values = self.data_points[:, 1]

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform DBScan.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color='#FF4C29', opacity=0.7))])
            fig.update_layout(
                title="DBSCAN Data",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color='#FF4C29', opacity=0.7))])

            fig.update_layout(
                title="DBSCAN Data",
                xaxis_title="X Values",
                yaxis_title="Y Values",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

            fig.update_xaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')
            fig.update_yaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')

        if kwargs.get('save'):
            fig.write_image('show_data.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def train(self) -> None:
        """Trains the Model"""
        self.clustering.fit(self.data_points)

    @property
    def labels(self):
        """Gets the labels"""
        return self.clustering.labels_

    def _get_outliers_2d(self):
        """Returns outliers from the data points separately during 2d config."""

        outliers_x = []
        outliers_y = []
        for label, xval, yval in list(zip(self.labels.squeeze(), self.x_values.squeeze(), self.y_values.squeeze())):
            if label != -1:
                continue
            outliers_x.append(xval)
            outliers_y.append(yval)

        return outliers_x, outliers_y

    def _get_outliers_3d(self):
        """Returns outliers from the data points separately during 3d config."""

        outliers_x1 = []
        outliers_x2 = []
        outliers_y = []
        for label, x1val, x2val, yval in list(zip(self.labels.squeeze(), self.x1_values.squeeze(),
                                                  self.x2_values.squeeze(), self.y_values.squeeze())):
            if label != -1:
                continue
            outliers_x1.append(x1val)
            outliers_x2.append(x2val)
            outliers_y.append(yval)

        return outliers_x1, outliers_x2, outliers_y

    def show_clusters(self, **kwargs) -> Figure:
        """
        Shows a plot of the clusters formed by K means.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            outliers_x1, outliers_x2, outliers_y = self._get_outliers_3d()

            fig = go.Figure(data=[go.Scatter3d(x=[x for x in self.x1_values.squeeze() if x not in outliers_x1],
                                               y=[x for x in self.x2_values.squeeze() if x not in outliers_x2],
                                               z=[x for x in self.y_values.squeeze() if x not in outliers_y],
                                               mode='markers',
                                               marker=dict(size=8,
                                                           color=[x + 5 for x in self.labels if x != -1],
                                                           # added 5 to avoid grey color
                                                           opacity=0.7),
                                               name='Data Points')])
            fig.add_traces(data=[go.Scatter3d(x=outliers_x1, y=outliers_x2, z=outliers_y, mode='markers',
                                              marker=dict(size=8, color='#FFFFFF'),
                                              name='Outliers')])

            fig.update_layout(
                title="Clustering",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            outliers_x, outliers_y = self._get_outliers_2d()
            fig = go.Figure(data=[go.Scatter(x=[x for x in self.x_values.squeeze() if x not in outliers_x],
                                             y=[x for x in self.y_values.squeeze() if x not in outliers_y],
                                             mode='markers',
                                             marker=dict(size=8,
                                                         color=[x + 5 for x in self.labels if x != -1],
                                                         # added 5 to avoid grey color
                                                         opacity=0.7),
                                             name='Data Points')])
            fig.add_traces(data=[go.Scatter(x=outliers_x, y=outliers_y, mode='markers',
                                            marker=dict(size=8, color='#FFFFFF'),
                                            name='Outliers')])

            fig.update_layout(
                title="Clustering",
                xaxis_title="X Values",
                yaxis_title="Y Values",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )
            fig.update_xaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')
            fig.update_yaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')

        if kwargs.get('save'):
            fig.write_image('show_clusters.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_silhouette_plot(self, **kwargs) -> Figure:
        """
        Shows a plot of the Silhouette Coefficient values of the data points.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        sample_silhouette_values = AllSilhouetteScores().compute(self.data_points, self.labels)
        scores_label = list(zip(sample_silhouette_values, self.labels))

        # Bug in mypy - cannot use lambda inside sort
        def _key0(x):
            return x[0]

        def _key1(x):
            return x[1]

        scores_label.sort(key=_key0, reverse=True)
        scores_label.sort(key=_key1)

        fig = go.Figure(data=[go.Bar(x=list(range(1, self.no_points + 1)), y=[x[0] for x in scores_label],
                                     marker=dict(color=[x[1] for x in scores_label]),
                                     name='Sample Silhouette Scores')])

        fig.update_layout(
            title="Silhouette Coefficient Values",
            xaxis_title="Number of Data Points",
            yaxis_title="Silhouette Score",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )
        fig.update_xaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')
        fig.update_yaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')

        if kwargs.get('save'):
            fig.write_image('show_silhouette_plot.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()
