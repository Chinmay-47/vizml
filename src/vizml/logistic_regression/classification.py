import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.linear_model import LogisticRegression as LogReg
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import (LinearlySeparable2DGenerator, LinearlySeparable3DGenerator,
                                  MoonData2DGenerator, MoonData3DGenerator,
                                  CircleDataGenerator, SphericalDataGenerator)


class LogisticRegression:
    """Class to perform and visualize Logistic Regression."""

    def __init__(self, no_points: int = 100, randomize: bool = False, random_state: int = -1,
                 is_3d: bool = False, data_shape: str = 'linearly_separable'):

        data_shape_generators = {(False, 'linearly_separable'): LinearlySeparable2DGenerator(random=randomize,
                                                                                             random_state=random_state),
                                 (True, 'linearly_separable'): LinearlySeparable3DGenerator(random=randomize,
                                                                                            random_state=random_state),
                                 (False, 'moon'): MoonData2DGenerator(random=randomize, random_state=random_state),
                                 (True, 'moon'): MoonData3DGenerator(random=randomize, random_state=random_state),
                                 (False, 'circle'): CircleDataGenerator(random=randomize, random_state=random_state),
                                 (True, 'circle'): SphericalDataGenerator(random=randomize, random_state=random_state)}

        self.no_points = no_points
        self.randomize = randomize
        self.is_3d = is_3d
        self.data_shape = data_shape

        dpgen = data_shape_generators[(self.is_3d, self.data_shape)]

        if self.is_3d:
            self.generated_data = dpgen.generate(no_of_points=self.no_points)
            self.x1_values = self.generated_data[:, 0]
            self.x2_values = self.generated_data[:, 1]
            self.y_values = self.generated_data[:, 2]
            self.labels = self.generated_data[:, 3]
            self.data_points = self.generated_data[:, :3]
        else:
            self.generated_data = dpgen.generate(no_of_points=self.no_points)
            self.x_values = self.generated_data[:, 0]
            self.y_values = self.generated_data[:, 1]
            self.labels = self.generated_data[:, 2]
            self.data_points = self.generated_data[:, :2]

        self.regressor = LogReg(n_jobs=-1)

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform Logistic Regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color=self.labels.squeeze(), opacity=0.7))])
            fig.update_layout(
                title="Logistic Regression Data",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.7))])

            fig.update_layout(
                title="Logistic Regression Data",
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
        self.regressor.fit(self.data_points, self.labels)

    @property
    def predicted_values(self):
        """Y-values predicted by the model"""
        return self.regressor.predict(self.data_points)

    def show_decision_boundary(self, **kwargs) -> Figure:
        """
        Shows a plot of the decision boundary formed by Logistic Regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            x_min, x_max = self.x1_values.min() - 1, self.x1_values.max() + 1
            y_min, y_max = self.x2_values.min() - 1, self.x2_values.max() + 1
            z_min, z_max = self.y_values.min() - 1, self.y_values.max() + 1
            x_range_vals = np.linspace(x_min, x_max, 200)
            y_range_vals = np.linspace(y_min, y_max, 200)
            z_range_vals = np.linspace(z_min, z_max, 200)

            xx, yy = np.meshgrid(x_range_vals, y_range_vals)
            zz, _ = np.meshgrid(z_range_vals, z_range_vals)

            Z = self.regressor.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color=self.labels, opacity=0.7),
                                               name='Data Points')])

            fig.add_traces(data=[go.Scatter3d(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), opacity=0.2,
                                              name='Decision Boundary', marker=dict(color=Z, opacity=0.2))])

            fig.update_layout(
                title="Classification",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            x_min, x_max = self.x_values.min() - 1, self.x_values.max() + 1
            y_min, y_max = self.y_values.min() - 1, self.y_values.max() + 1
            x_range_vals = np.linspace(x_min, x_max, 200)
            y_range_vals = np.linspace(y_min, y_max, 200)
            xx, yy = np.meshgrid(x_range_vals, y_range_vals)
            Z = self.regressor.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.7),
                                             name='Data Points')])

            fig.add_traces(data=[go.Contour(x=x_range_vals, y=y_range_vals, z=Z, connectgaps=True,
                                            opacity=0.2, name='Decision Boundary', showscale=False)])

            fig.update_layout(
                title="Classification",
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

    def show_decision_probabilities(self, **kwargs) -> Figure:
        """
        Shows a plot of the decision probabilities formed by Logistic Regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            x_min, x_max = self.x1_values.min() - 1, self.x1_values.max() + 1
            y_min, y_max = self.x2_values.min() - 1, self.x2_values.max() + 1
            z_min, z_max = self.y_values.min() - 1, self.y_values.max() + 1
            x_range_vals = np.linspace(x_min, x_max, 200)
            y_range_vals = np.linspace(y_min, y_max, 200)
            z_range_vals = np.linspace(z_min, z_max, 200)

            xx, yy = np.meshgrid(x_range_vals, y_range_vals)
            zz, _ = np.meshgrid(z_range_vals, z_range_vals)

            Z = self.regressor.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

            Z = (Z - Z.mean()) / (Z.max() - Z.min())

            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers', showlegend=False,
                                               marker=dict(size=8, color=self.labels, opacity=0.7))])

            fig.add_traces(data=[go.Scatter3d(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), opacity=0.2,
                                              marker=dict(color=Z, opacity=0.2, showscale=True),
                                              showlegend=False)])

            fig.update_layout(
                title="Decision Probabilities",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            x_min, x_max = self.x_values.min() - 1, self.x_values.max() + 1
            y_min, y_max = self.y_values.min() - 1, self.y_values.max() + 1
            x_range_vals = np.linspace(x_min, x_max, 200)
            y_range_vals = np.linspace(y_min, y_max, 200)
            xx, yy = np.meshgrid(x_range_vals, y_range_vals)
            Z = self.regressor.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            Z = (Z - Z.mean()) / (Z.max() - Z.min())

            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.7),
                                             showlegend=False)])

            fig.add_traces(data=[go.Contour(x=x_range_vals, y=y_range_vals, z=Z, connectgaps=True,
                                            opacity=0.5, showscale=True, showlegend=False)])

            fig.update_layout(
                title="Classification",
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
