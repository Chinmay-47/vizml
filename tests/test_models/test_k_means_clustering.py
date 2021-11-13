from numpy import equal
from plotly.graph_objects import Figure
from vizml.k_means_clustering.clustering import KMeansClustering


def test_show_data():
    """Tests the show data function in K means Clustering."""

    clu = KMeansClustering()
    fig = clu.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_clusters():
    """Tests the show clusters function in K means Clustering."""

    clu = KMeansClustering()
    clu.train()
    fig = clu.show_clusters(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_elbow_plot():
    """Tests the show elbow plot function in K means Clustering."""

    clu = KMeansClustering()
    fig = clu.show_elbow_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_3d():
    """Tests the show data function in K means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    fig = clu.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_clusters_3d():
    """Tests the show clusters function in K means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    clu.train()
    fig = clu.show_clusters(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_elbow_plot_3d():
    """Tests the show elbow plot function in K means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    fig = clu.show_elbow_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize():
    """Tests the random initialization of K means Clustering."""

    clu1 = KMeansClustering()
    clu2 = KMeansClustering(randomize=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in K means Clustering."""

    clu1 = KMeansClustering(random_state=7)
    clu2 = KMeansClustering(random_state=7)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d():
    """Tests the random initialization of K means Clustering for 3d config."""

    clu1 = KMeansClustering(is_3d=True)
    clu2 = KMeansClustering(randomize=True, is_3d=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d():
    """Tests the initialization with a fixed random state in K means Clustering for 3d config."""

    clu1 = KMeansClustering(random_state=7, is_3d=True)
    clu2 = KMeansClustering(random_state=7, is_3d=True)
    data1 = clu1.data_points
    data2 = clu2.data_points

    assert equal(data1, data2).all()
