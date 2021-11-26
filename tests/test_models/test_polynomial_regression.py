from numpy import equal
from plotly.graph_objects import Figure
from vizml.polynomial_regression.regression import PolynomialRegression


def test_show_data():
    """Tests the show data function in Polynomial Regression."""

    reg = PolynomialRegression()
    fig = reg.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_curve():
    """Tests the show regression curve function in Polynomial Regression."""

    reg = PolynomialRegression()
    reg.train()
    fig = reg.show_regression_curve(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores():
    """Tests the show error scores function in Polynomial Regression."""

    reg = PolynomialRegression()
    reg.train()
    fig = reg.show_error_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize():
    """Tests the random initialization of Polynomial Regression."""

    reg1 = PolynomialRegression()
    reg2 = PolynomialRegression(randomize=True)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in Polynomial Regression."""

    reg1 = PolynomialRegression(random_state=7)
    reg2 = PolynomialRegression(random_state=7)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert equal(data1, data2).all()
