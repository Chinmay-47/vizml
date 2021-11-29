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


def test_show_avg_silhouette_scores():
    """Tests the show avg silhouette scores function in K Means Clustering."""

    clu = KMeansClustering()
    fig = clu.show_avg_silhouette_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_silhouette_plot():
    """Tests the show silhouette plot function in K Means Clustering."""

    clu = KMeansClustering()
    clu.train()
    fig = clu.show_silhouette_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_show_freq_distribution():
    """Tests the show frequency distribution function in K Means Clustering."""

    clu = KMeansClustering()
    clu.train()
    fig = clu.show_freq_distribution(return_fig=True)
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


def test_show_avg_silhouette_scores_3d():
    """Tests the show avg silhouette scores function in K Means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    fig = clu.show_avg_silhouette_scores(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_silhouette_plot_3d():
    """Tests the show silhouette plot function in K Means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    clu.train()
    fig = clu.show_silhouette_plot(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_show_freq_distribution_3d():
    """Tests the show frequency distribution function in K Means Clustering for 3d config."""

    clu = KMeansClustering(is_3d=True)
    clu.train()
    fig = clu.show_freq_distribution(return_fig=True)
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


def test_change_num_clusters():
    """Tests the function to change number of clusters to detect."""

    clu1 = KMeansClustering(no_clusters=3)
    clu1.change_num_clusters(5)

    assert clu1.no_clusters == 5 and clu1.clustering.n_clusters == 5


def test_change_num_clusters_3d():
    """Tests the function to change number of clusters to detect for 3d config."""

    clu1 = KMeansClustering(no_clusters=3, is_3d=True)
    clu1.change_num_clusters(5)

    assert clu1.no_clusters == 5 and clu1.clustering.n_clusters == 5
