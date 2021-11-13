import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.cluster import KMeans
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import Normal2DGenerator, Normal3DGenerator, NormalDataGenerator


class KMeansClustering:
    """Class to perform and visualize K Means Clustering."""

    def __init__(self, no_points: int = 60, no_clusters: int = 3, randomize: bool = False,
                 random_state: int = -1, is_3d: bool = False):

        self.no_clusters = no_clusters
        self.clustering = KMeans(n_clusters=no_clusters)
        self.randomize = randomize
        self.is_3d = is_3d
        dpgen: NormalDataGenerator
        if self.is_3d:
            dpgen = Normal3DGenerator(random=randomize, random_state=random_state)
            self.data_points = dpgen.generate(no_of_points=no_points)
            self.x1_values = self.data_points[:, 0]
            self.x2_values = self.data_points[:, 1]
            self.y_values = self.data_points[:, 2]
        else:
            dpgen = Normal2DGenerator(random=randomize, random_state=random_state)
            self.data_points = dpgen.generate(no_of_points=no_points)
            self.x_values = self.data_points[:, 0]
            self.y_values = self.data_points[:, 1]

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
                                              marker=dict(size=8, color='#FFFFFF'),
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
                                            marker=dict(size=8, color='#FFFFFF'), name='Cluster Centers')])

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
        for i in range(1, 16):
            k_means = KMeans(n_clusters=i)
            k_means.fit(self.data_points)
            inertia = k_means.inertia_
            wcss_list.append(inertia)

        fig = go.Figure(data=[go.Scatter(x=list(range(1, 16)), y=wcss_list,
                                         marker=dict(color='#6D9886'), name='Elbow Plot')])

        fig.add_traces(data=[go.Scatter(x=[self.no_clusters],
                                        y=[wcss_list[self.no_clusters - 1]], mode='markers',
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
