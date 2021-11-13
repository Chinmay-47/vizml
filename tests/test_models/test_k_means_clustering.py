from plotly.graph_objects import Figure
from vizml.k_means_clustering.clustering import KMeansClustering


def test_show_data():
    """Tests the show data function in K means Clustering"""

    clu = KMeansClustering()
    fig = clu.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_clusters():
    """Tests the show clusters function in K means Clustering"""

    clu = KMeansClustering()
    clu.train()
    fig = clu.show_clusters(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_elbow_plot():
    """Tests the show elbow plot function in K means Clustering"""

    clu = KMeansClustering()
    fig = clu.show_elbow_plot(return_fig=True)
    assert isinstance(fig, Figure)
