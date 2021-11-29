import pytest
from numpy import equal
from plotly.graph_objects import Figure
from vizml.dbscan.clustering import DBScan


def test_show_data():
    """Tests the show data function in DBScan."""

    clu = DBScan()
    fig = clu.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_clusters():
    """Tests the show clusters function in DBScan."""

    clu = DBScan()
    clu.train()
    fig = clu.show_clusters(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics():
    """Tests the show metrics function in DBScan."""

    clu = DBScan()
    clu.train()
    fig = clu.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_silhouette_plot():
    """Tests the show silhouette plot function in DBScan."""

    clu = DBScan()
    clu.train()
    fig = clu.show_silhouette_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_show_freq_distribution():
    """Tests the show frequency distribution function in DBScan."""

    clu = DBScan()
    clu.train()
    fig = clu.show_freq_distribution(return_fig=True)
    assert isinstance(fig, Figure)


def test_avg_silhouette_score():
    """Tests the average silhouette score property in DBScan."""

    clu = DBScan()
    clu.train()
    avg_score = clu.avg_silhouette_score
    assert avg_score == 0.02


def test_show_data_3d():
    """Tests the show data function in DBScan for 3d config."""

    clu = DBScan(is_3d=True)
    fig = clu.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_clusters_3d():
    """Tests the show clusters function in DBScan for 3d config."""

    clu = DBScan(is_3d=True)
    clu.train()
    fig = clu.show_clusters(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics_3d():
    """Tests the show metrics function in DBScan for 3d config."""

    clu = DBScan(is_3d=True)
    clu.train()
    fig = clu.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_silhouette_plot_3d():
    """Tests the show silhouette plot function in DBScan for 3d config."""

    clu = DBScan(is_3d=True)
    clu.train()
    fig = clu.show_silhouette_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_show_freq_distribution_3d():
    """Tests the show frequency distribution function in DBScan for 3d config."""

    clu = DBScan(is_3d=True)
    clu.train()
    fig = clu.show_freq_distribution(return_fig=True)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize(
    "test_vals", [(0.5, 1.0), (0.7, -0.09), (1.0, 0.23)]
)
def test_avg_silhouette_score_3d(test_vals):
    """Tests the average silhouette score property in DBScan for 3d config."""

    _min_dist, avg_sil_score = test_vals
    clu = DBScan(is_3d=True, min_dist=_min_dist)
    clu.train()
    assert avg_sil_score == clu.avg_silhouette_score


def test_randomize():
    """Tests the random initialization of DBScan."""

    clu1 = DBScan()
    clu2 = DBScan(randomize=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in DBScan."""

    clu1 = DBScan(random_state=7)
    clu2 = DBScan(random_state=7)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d():
    """Tests the random initialization of DBScan for 3d config."""

    clu1 = DBScan(is_3d=True)
    clu2 = DBScan(randomize=True, is_3d=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d():
    """Tests the initialization with a fixed random state in DBScan for 3d config."""

    clu1 = DBScan(random_state=7, is_3d=True)
    clu2 = DBScan(random_state=7, is_3d=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert equal(data1, data2).all()
