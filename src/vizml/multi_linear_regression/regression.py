from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import Linear1DGenerator, Linear2DGenerator
from vizml.metrics.regression_metrics import compute_all_errors


class MultiLinearRegression:
    """
    Performs and Visualizes Multi Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 random_state: int = -1):
        self.regressor = LinearRegression(n_jobs=-1)
        self.randomize = randomize
        dpgen1 = Linear2DGenerator(random=randomize, random_state=random_state)
        dpgen2 = Linear1DGenerator(random=randomize, random_state=random_state)
        self.x_values = dpgen1.generate(no_of_points=no_points)
        self.y_values = dpgen2.generate(no_of_points=no_points, is_increasing=is_increasing)
        self.data_points: Any = np.concatenate((self.x_values, self.y_values), axis=1)

    def train(self) -> None:
        """Trains the Model"""
        self.regressor.fit(self.x_values, self.y_values)

    @property
    def predicted_values(self):
        """Y-values predicted by the model"""
        return self.regressor.predict(self.x_values)

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform multi linear regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        x1 = self.x_values[:, 0]
        x2 = self.x_values[:, 1]

        fig = go.Figure(data=[go.Scatter3d(x=x1, y=x2, z=self.y_values.squeeze(), mode='markers',
                                           marker=dict(size=8, color='#FF4C29', opacity=0.7))])
        fig.update_layout(
            title="Multi Linear Regression Data",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )

        if kwargs.get('save'):
            fig.write_image('show_data.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_regression_plane(self, **kwargs) -> Figure:
        """
        Shows a plot of the current regression plane with data.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        x1 = self.x_values[:, 0]
        x2 = self.x_values[:, 1]

        fig = go.Figure(data=[go.Scatter3d(x=x1, y=x2, z=self.y_values.squeeze(), mode='markers',
                                           marker=dict(size=8, color='#FF4C29', opacity=0.7))])

        fig.add_traces(data=[go.Mesh3d(x=x1,
                                       y=x2,
                                       z=self.predicted_values.squeeze(),
                                       name='Regression Plane',
                                       color='#6D9886')])

        fig.update_layout(
            title="Regression Plane",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )

        if kwargs.get('save'):
            fig.write_image('show_regression_plane.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_error_scores(self, **kwargs) -> Figure:
        """
        Shows a plot of the different error metrics for current regression plane.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        computed_errors = compute_all_errors(self.y_values, self.predicted_values)
        err_types, err_metrics = tuple(zip(*computed_errors))

        fig = go.Figure(data=[go.Bar(x=err_metrics, y=err_types, text=err_metrics, textposition='inside',
                                     orientation='h', marker=dict(color='#FF4C29', opacity=0.6))])

        fig.update_layout(
            title="Error Metrics Computed",
            xaxis_title="Error Value",
            yaxis_title="Error Metric",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )

        fig.update_yaxes(type='category')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        if kwargs.get('save'):
            fig.write_image('show_error_scores.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()


class OrdinaryLeastSquaresRegression(MultiLinearRegression):
    """
    Performs and Visualizes Ordinary Least Multi Linear Regression.
    """


class LassoRegression(MultiLinearRegression):
    """
    Performs and Visualizes Lasso Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 l1_penalty: float = 1.0, random_state: int = -1):
        super().__init__(no_points=no_points, is_increasing=is_increasing, randomize=randomize,
                         random_state=random_state)
        self.regressor = Lasso(alpha=l1_penalty)


class RidgeRegression(MultiLinearRegression):
    """
    Performs and Visualizes Ridge Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 l2_penalty: float = 1.0, random_state: int = -1):
        super().__init__(no_points=no_points, is_increasing=is_increasing, randomize=randomize,
                         random_state=random_state)
        self.regressor = Ridge(alpha=l2_penalty)
