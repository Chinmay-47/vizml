from collections import Counter

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.cluster import KMeans
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import Normal2DGenerator, Normal3DGenerator, NormalDataGenerator
from vizml.metrics.clustering_metrics import AvgSilhouetteScore, AllSilhouetteScores


class KMeansClustering:
    """Class to perform and visualize K Means Clustering."""

    def __init__(self, no_points: int = 100, no_clusters: int = 3, randomize: bool = False,
                 random_state: int = -1, is_3d: bool = False):

        self.no_clusters = no_clusters
        self.clustering = KMeans(n_clusters=no_clusters)
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

    def change_num_clusters(self, no_clusters: int) -> None:
        """Change the number of clusters to detect for the same data."""
        self.no_clusters = no_clusters
        self.clustering = KMeans(n_clusters=no_clusters)

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform K means clustering.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color='#FF4C29', opacity=0.7))])
            fig.update_layout(
                title="K Means Clustering Data",
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
                title="K Means Clustering Data",
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

    def show_clusters(self, **kwargs) -> Figure:
        """
        Shows a plot of the clusters formed by K means.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color=self.labels, opacity=0.7),
                                               name='Data Points')])
            fig.add_traces(data=[go.Scatter3d(x=self.clustering.cluster_centers_[:, 0],
                                              y=self.clustering.cluster_centers_[:, 1],
                                              z=self.clustering.cluster_centers_[:, 2], mode='markers',
                                              marker=dict(size=6, color='#FFFFFF'),
                                              name='Cluster Centers')])
            fig.update_layout(
                title="Clustering",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels, opacity=0.7),
                                             name='Data Points')])

            fig.add_traces(data=[go.Scatter(x=self.clustering.cluster_centers_[:, 0],
                                            y=self.clustering.cluster_centers_[:, 1], mode='markers',
                                            marker=dict(size=6, color='#FFFFFF'), name='Cluster Centers')])

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

    def show_elbow_plot(self, **kwargs) -> Figure:
        """
        Shows a plot of the no_clusters vs Within Cluster Sum of Squares (WCSS).

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        wcss_list = []
        for i in range(2, 11):
            k_means = KMeans(n_clusters=i)
            k_means.fit(self.data_points)
            inertia = k_means.inertia_
            wcss_list.append(inertia)

        fig = go.Figure(data=[go.Scatter(x=list(range(2, 11)), y=wcss_list,
                                         marker=dict(color='#6D9886'), name='Elbow Plot')])

        fig.add_traces(data=[go.Scatter(x=[self.no_clusters],
                                        y=[wcss_list[self.no_clusters - 2]], mode='markers',
                                        marker=dict(size=8, color='#FFFFFF'),
                                        name='Current Clusters')])

        fig.update_layout(
            title="Elbow Method",
            xaxis_title="Number of Clusters",
            yaxis_title="WCSS",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )
        fig.update_xaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')
        fig.update_yaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')

        if kwargs.get('save'):
            fig.write_image('show_elbow_method_plot.jpeg')

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

    def show_avg_silhouette_scores(self, **kwargs) -> Figure:
        """
        Shows a plot of the no_clusters vs Average Silhouette Scores.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        silhouette_scores = []
        silhouette_scores_rounded = []
        for i in range(2, 11):
            k_means = KMeans(n_clusters=i)
            k_means.fit(self.data_points)
            silhouette_score = AvgSilhouetteScore().compute(self.data_points, k_means.labels_)
            silhouette_scores.append(silhouette_score)
            silhouette_scores_rounded.append(round(silhouette_score, 3))

        fig = go.Figure(data=[go.Scatter(x=list(range(2, 11)), y=silhouette_scores,
                                         marker=dict(color='#6D9886'), name='Average Silhouette Scores')])

        fig.update_layout(
            title="Average Silhouette Scores",
            xaxis_title="Number of Clusters",
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
            fig.write_image('show_avg_silhouette_scores.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_freq_distribution(self, **kwargs) -> Figure:
        """
        Shows a plot of the frequency distribution of the data points in each cluster.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        _x, _y = list(zip(*Counter(self.labels).items()))

        fig = go.Figure(data=[go.Bar(x=_x, y=_y,
                                     text=_y, textposition='inside',
                                     marker=dict(color='#FF4C29', opacity=0.7))])

        fig.update_layout(
            title="Data Points in Each Cluster",
            title_x=0.5,
            xaxis_title="Labels",
            yaxis_title="Frequency",
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )
        fig.update_xaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')
        fig.update_yaxes(gridcolor='#000000', zerolinewidth=2, zerolinecolor='#000000')

        if kwargs.get('save'):
            fig.write_image('show_freq_distribution.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()
