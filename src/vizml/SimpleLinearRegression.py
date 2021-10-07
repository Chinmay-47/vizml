import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.linear_model import LinearRegression
from vizml.data_generator import Linear1DGenerator


class SimpleLinearRegression:
    """
    Performs and Visualizes Simple Linear Regression.
    """

    def __init__(self, no_points: int = 20, is_increasing: bool = True, randomize: bool = False):
        self.regressor = LinearRegression(n_jobs=-1)
        self.randomize = randomize
        dpgen = Linear1DGenerator(random=randomize)
        self.x_values = dpgen.generate(no_of_points=no_points, is_increasing=is_increasing)
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

        fig = go.Figure(data=[go.Scatter(x=self.x_values[:, 0], y=self.y_values[:, 0], mode='markers',
                                         marker=dict(size=8, color='red', opacity=0.8))])
        fig.update_layout(
            title="Simple Linear Regression Data",
            xaxis_title="X Values",
            yaxis_title="Y Values",
            title_x=0.5
        )

        if 'save' in kwargs and kwargs['save']:
            fig.write_image('show_data.jpeg')

        if 'return_fig' in kwargs and kwargs['return_fig']:
            return fig

        fig.show()
