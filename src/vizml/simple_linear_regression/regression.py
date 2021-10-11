import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from vizml.data_generator import Linear1DGenerator
from vizml.error_metrics import compute_all_errors


class SimpleLinearRegression:
    """
    Performs and Visualizes Simple Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False):
        self.regressor = LinearRegression(n_jobs=-1)
        self.randomize = randomize
        dpgen = Linear1DGenerator(random=randomize)
        self.x_values = dpgen.generate(no_of_points=no_points)
        self.y_values = dpgen.generate(no_of_points=no_points, is_increasing=is_increasing)

    def train(self):
        """Trains the Model"""
        self.regressor.fit(self.x_values, self.y_values)

    @property
    def predicted_values(self):
        """Y-values predicted by the model"""
        return self.regressor.predict(self.x_values)

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform linear regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                         marker=dict(size=8, color='#FF4C29', opacity=0.7))])
        fig.update_layout(
            title="Simple Linear Regression Data",
            xaxis_title="X Values",
            yaxis_title="Y Values",
            title_x=0.5,
            plot_bgcolor='#212121',
            paper_bgcolor='#212121',
            # font_color='#7FDBFF'
        )
        fig.update_xaxes(gridcolor='#035397', gridwidth=1)
        fig.update_yaxes(gridcolor='#035397', gridwidth=1)

        if kwargs.get('save'):
            fig.write_image('show_data.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_regression_line(self, **kwargs) -> Figure:
        """
        Shows a plot of the current regression line with data.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                         marker=dict(size=8, color='#FF4C29', opacity=0.7), name='Data Points')])

        fig.add_traces(data=[go.Scatter(x=self.x_values.squeeze(),
                                        y=self.predicted_values.squeeze(),
                                        name='Regression Line', marker=dict(color='#6D9886'))])

        fig.update_layout(
            title="Regression Line",
            xaxis_title="X Values",
            yaxis_title="Y Values",
            title_x=0.5,
            plot_bgcolor='#212121',
            paper_bgcolor='#212121',
            # font_color='#7FDBFF'
        )
        fig.update_xaxes(gridcolor='#F9F3DF')
        fig.update_yaxes(gridcolor='#F9F3DF')

        if kwargs.get('save'):
            fig.write_image('show_regression_line.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_error_scores(self, **kwargs) -> Figure:
        """
        Shows a plot of the different error metrics for current regression line.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        computed_errors = compute_all_errors(self.y_values, self.predicted_values)
        err_types, err_metrics = tuple(zip(*computed_errors))

        fig = go.Figure(data=[go.Bar(x=err_metrics, y=err_types, orientation='h',
                                     marker=dict(color='#FF4C29', opacity=0.6))])

        fig.update_layout(
            title="Error Metrics Computed",
            xaxis_title="Error Value",
            yaxis_title="Error Metric",
            title_x=0.5,
            plot_bgcolor='#212121',
            paper_bgcolor='#212121',
            # font_color='#7FDBFF'
        )

        fig.update_yaxes(type='category')
        fig.update_xaxes(gridcolor='#F9F3DF')
        fig.update_yaxes(gridcolor='#F9F3DF')

        if kwargs.get('save'):
            fig.write_image('show_error_scores.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()


class OrdinaryLeastSquaresRegression(SimpleLinearRegression):
    """
    Performs and Visualizes Ordinary Least Squares Linear Regression.
    """


class LassoRegression(SimpleLinearRegression):
    """
    Performs and Visualizes Lasso Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 l1_penalty: float = 0.1):
        super().__init__(no_points=no_points, is_increasing=is_increasing, randomize=randomize)
        self.regressor = Lasso(alpha=l1_penalty)


class RidgeRegression(SimpleLinearRegression):
    """
    Performs and Visualizes Ridge Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False,
                 l2_penalty: float = 0.1):
        super().__init__(no_points=no_points, is_increasing=is_increasing, randomize=randomize)
        self.regressor = Ridge(alpha=l2_penalty)
