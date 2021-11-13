from numpy import equal
from plotly.graph_objects import Figure
from vizml.multi_linear_regression.regression import (OrdinaryLeastSquaresRegression,
                                                      LassoRegression, RidgeRegression)


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


def test_show_regression_plane_OLS():
    """Tests the show regression plane function in OLS."""

    reg = OrdinaryLeastSquaresRegression()
    reg.train()
    fig = reg.show_regression_plane(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_plane_Lasso():
    """Tests the show regression plane function in Lasso."""

    reg = LassoRegression()
    reg.train()
    fig = reg.show_regression_plane(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_regression_plane_Ridge():
    """Tests the show regression plane function in Ridge."""

    reg = RidgeRegression()
    reg.train()
    fig = reg.show_regression_plane(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_OLS():
    """Tests the show error scores function in OLS."""

    reg = OrdinaryLeastSquaresRegression()
    reg.train()
    fig = reg.show_error_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_Lasso():
    """Tests the show error scores function in Lasso."""

    reg = LassoRegression()
    reg.train()
    fig = reg.show_error_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_error_scores_Ridge():
    """Tests the show error scores function in Ridge."""

    reg = RidgeRegression()
    reg.train()
    fig = reg.show_error_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize_OLS():
    """Tests the random initialization of OLS."""

    reg1 = OrdinaryLeastSquaresRegression()
    reg2 = OrdinaryLeastSquaresRegression(randomize=True)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert not equal(data1, data2).any()


def test_randomize_Lasso():
    """Tests the random initialization of Lasso."""

    reg1 = LassoRegression()
    reg2 = LassoRegression(randomize=True)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert not equal(data1, data2).any()


def test_randomize_Ridge():
    """Tests the random initialization of Ridge."""

    reg1 = RidgeRegression()
    reg2 = RidgeRegression(randomize=True)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert not equal(data1, data2).any()


def test_random_state_OLS():
    """Tests the initialization with a fixed random state in OLS."""

    reg1 = OrdinaryLeastSquaresRegression(random_state=7)
    reg2 = OrdinaryLeastSquaresRegression(random_state=7)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert equal(data1, data2).all()


def test_random_state_Lasso():
    """Tests the initialization with a fixed random state in Lasso."""

    reg1 = LassoRegression(random_state=7)
    reg2 = LassoRegression(random_state=7)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert equal(data1, data2).all()


def test_random_state_Ridge():
    """Tests the initialization with a fixed random state in Ridge."""

    reg1 = RidgeRegression(random_state=7)
    reg2 = RidgeRegression(random_state=7)
    data1 = reg1.data_points
    data2 = reg2.data_points

    assert equal(data1, data2).all()
