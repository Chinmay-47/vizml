from plotly.graph_objects import Figure
from vizml.SimpleLinearRegression import OrdinaryLeastSquaresRegression, LassoRegression, RidgeRegression


def test_show_data_OLS():
    """Tests the show data function in OLS."""

    reg = OrdinaryLeastSquaresRegression()
    fig = reg.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_Lasso():
    """Tests the show data function in Lasso."""

    reg = LassoRegression()
    fig = reg.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_Ridge():
    """Tests the show data function in Ridge."""

    reg = RidgeRegression()
    fig = reg.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_line_OLS():
    """Tests the show regression line function in OLS."""

    reg = OrdinaryLeastSquaresRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_line_Lasso():
    """Tests the show regression line function in Lasso."""

    reg = LassoRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_line_Ridge():
    """Tests the show regression line function in Ridge."""

    reg = RidgeRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_OLS():
    """Tests the show error scores function in OLS."""

    reg = OrdinaryLeastSquaresRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_Lasso():
    """Tests the show error scores function in Lasso."""

    reg = LassoRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_Ridge():
    """Tests the show error scores function in Ridge."""

    reg = RidgeRegression()
    reg.train()
    fig = reg.show_regression_line(return_fig=True)
    assert isinstance(fig, Figure)
