from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.graph_objects import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import Linear1DGenerator
from vizml.metrics.regression_metrics import compute_all_errors


class PolynomialRegression:
    """
    Performs and Visualizes Polynomial Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 random_state: int = -1, degree: int = 2):

        self.regressor = LinearRegression(n_jobs=-1)
        self.randomize = randomize
        self.degree = degree
        self.no_points = no_points
        self.is_increasing = is_increasing

        dpgen = Linear1DGenerator(random=randomize, random_state=random_state)
        self.x_values = dpgen.generate(no_of_points=no_points)
        self.y_values = dpgen.generate(no_of_points=no_points, is_increasing=is_increasing)
        self.data_points = np.concatenate((self.x_values, self.y_values), axis=1)

        poly_reg = PolynomialFeatures(degree=self.degree)
        poly_reg.fit(self.x_values)

        self.X_poly = poly_reg.transform(self.x_values)
        self.x_range = np.linspace(self.x_values.min(), self.x_values.max(), 100).reshape(-1, 1)
        self.X_poly_range = poly_reg.transform(self.x_range)

    def train(self) -> None:
        """Trains the Model"""
        self.regressor.fit(self.X_poly, self.y_values)

    def _predicted_vals_for_plot(self):
        """Y-values predicted by model used for plotting."""
        return self.regressor.predict(self.X_poly_range)

    @property
    def predicted_values(self):
        """Y-values predicted by the model"""
        return self.regressor.predict(self.X_poly)

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform polynomial regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                         marker=dict(size=8, color='#FF4C29', opacity=0.7))])

        fig.update_layout(
            title="Polynomial Regression Data",
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

    @staticmethod
    def _format_coeff(coeffs: NDArray[Any]) -> str:
        """Utility function to get the equation of the polynomial regression."""

        equation_list = [f"{coeff}x^{i}" for i, coeff in enumerate(coeffs)]
        equation = "$" + " + ".join(equation_list) + "$"
        replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
        for old, new in replace_map.items():
            equation = equation.replace(old, new)

        return equation

    @property
    def equation(self):
        return self._format_coeff(self.regressor.coef_.round(2)[0])

    def show_regression_curve(self, **kwargs) -> Figure:
        """
        Shows a plot of the current regression curve with data.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                         marker=dict(size=8, color='#FF4C29', opacity=0.7), name='Data Points')])

        fig.add_traces(data=[go.Scatter(x=self.x_range.squeeze(),
                                        y=self._predicted_vals_for_plot().squeeze(),
                                        name="Regression Curve", marker=dict(color='#6D9886'))])

        fig.update_layout(
            title=self.equation,
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
            fig.write_image('show_regression_curve.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_error_scores(self, **kwargs) -> Figure:
        """
        Shows a plot of the different error metrics for current regression curve.

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
