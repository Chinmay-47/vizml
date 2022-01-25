import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.naive_bayes import GaussianNB
from vizml._dashboard_configs import DASH_STYLE, PLOT_TEMPLATE
from vizml.data_generator import (LinearlySeparable2DGenerator, LinearlySeparable3DGenerator,
                                  MoonData2DGenerator, MoonData3DGenerator,
                                  CircleDataGenerator, SphericalDataGenerator)
from vizml.metrics.classification_metrics import compute_all_metrics, compute_all_prob_metrics, ConfusionMatrix


class NaiveBayes:
    """Class to perform Classification and visualize Naive Bayes."""

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

        self.classifier = GaussianNB()

    def show_data(self, **kwargs) -> Figure:
        """
        Shows a plot of the data points used to perform Classification.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        if self.is_3d:
            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color=self.labels.squeeze(), opacity=0.8))])
            fig.update_layout(
                title="Naive Bayes Data",
                title_x=0.5,
                plot_bgcolor=DASH_STYLE["backgroundColor"],
                paper_bgcolor=DASH_STYLE["backgroundColor"],
                font_color=DASH_STYLE["color"],
                template=PLOT_TEMPLATE
            )

        else:
            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.8))])

            fig.update_layout(
                title="Naive Bayes Data",
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
        self.classifier.fit(self.data_points, self.labels)

    @property
    def predicted_values(self):
        """Labels predicted by the model"""
        return self.classifier.predict(self.data_points)

    @property
    def decision_function(self):
        """Decision Probabilities predicted by the model."""
        return self.classifier.predict_proba(self.data_points)[:, 1]

    def show_decision_boundary(self, **kwargs) -> Figure:
        """
        Shows a plot of the decision boundary formed by Naive Bayes classifier.

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

            Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers',
                                               marker=dict(size=8, color=self.labels, opacity=0.8),
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
            Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.8),
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
            fig.write_image('show_decision_boundary.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_decision_probabilities(self, **kwargs) -> Figure:
        """
        Shows a plot of the decision probabilities formed by Naive Bayes classifier.

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

            Z = self.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]

            Z = (Z - Z.mean()) / (Z.max() - Z.min())

            fig = go.Figure(data=[go.Scatter3d(x=self.x1_values.squeeze(), y=self.x2_values.squeeze(),
                                               z=self.y_values.squeeze(), mode='markers', showlegend=False,
                                               marker=dict(size=8, color=self.labels, opacity=0.8))])

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
            x_range_vals = np.linspace(x_min, x_max, 300)
            y_range_vals = np.linspace(y_min, y_max, 300)
            xx, yy = np.meshgrid(x_range_vals, y_range_vals)
            Z = self.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
            Z = (Z - Z.mean()) / (Z.max() - Z.min())

            fig = go.Figure(data=[go.Scatter(x=self.x_values.squeeze(), y=self.y_values.squeeze(), mode='markers',
                                             marker=dict(size=8, color=self.labels.squeeze(), opacity=0.8),
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
            fig.write_image('show_decision_probabilities.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_confusion_matrix(self, **kwargs) -> Figure:
        """
        Shows a plot of the Confusion Matrix for current classifier.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        tn, fp, fn, tp = ConfusionMatrix().compute(self.labels, self.predicted_values).ravel()
        z = [[fp, tn], [tp, fn]]

        fig = go.Figure(data=[go.Heatmap(x=["P", "N"], y=["N", "P"], z=z, opacity=0.7,
                                         text=z, texttemplate="%{z}", colorscale='Blackbody')])

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Labels",
            yaxis_title="Actual Labels",
            title_x=0.5,
            plot_bgcolor=DASH_STYLE["backgroundColor"],
            paper_bgcolor=DASH_STYLE["backgroundColor"],
            font_color=DASH_STYLE["color"],
            template=PLOT_TEMPLATE
        )

        fig.update_yaxes(type='category')
        fig.update_xaxes(type='category')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        if kwargs.get('save'):
            fig.write_image('show_confusion_matrix.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()

    def show_metrics(self, **kwargs) -> Figure:
        """
        Shows a plot of the different metrics for current classifier.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        computed_metrics = compute_all_metrics(self.labels, self.predicted_values)
        metric_type, metric_val = tuple(zip(*computed_metrics))

        computed_prob_metrics = compute_all_prob_metrics(self.labels, self.decision_function)
        prob_metric_type, prob_metric_val = tuple(zip(*computed_prob_metrics))

        metric_type += prob_metric_type
        metric_val += prob_metric_val

        metric_type = tuple(reversed(metric_type))
        metric_val = tuple(reversed(metric_val))

        fig = go.Figure(data=[go.Bar(x=metric_val, y=metric_type, text=metric_val, textposition='inside',
                                     orientation='h', marker=dict(color='#FF4C29', opacity=0.6))])

        fig.update_layout(
            title="Classification Metrics Computed",
            xaxis_title="Metric Value",
            yaxis_title="Metric Name",
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
            fig.write_image('show_metrics.jpeg')

        if kwargs.get('return_fig'):
            return fig

        fig.show()
