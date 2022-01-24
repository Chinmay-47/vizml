from numpy import equal, ndarray
from plotly.graph_objects import Figure
from vizml.k_nearest_neighbours.classification import KNearestNeighbours


def test_show_data():
    """Tests the show data function in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary():
    """Tests the show decision boundary function in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(no_points=10, k_neighbors=2)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities():
    """Tests the show decision probabilities function in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(no_points=10, k_neighbors=2)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics():
    """Tests the show metrics function in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix():
    """Tests the show confusion matrix function in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_3d():
    """Tests the show data function in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True, no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary_3d():
    """Tests the show decision boundary function in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True, no_points=10, k_neighbors=2)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities_3d():
    """Tests the show decision probabilities function in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True, no_points=10, k_neighbors=2)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics_3d():
    """Tests the show metrics function in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix_3d():
    """Tests the show silhouette plot function in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize():
    """Tests the random initialization of K Nearest Neighbours classifier."""

    clf1 = KNearestNeighbours()
    clf2 = KNearestNeighbours(randomize=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier."""

    clf1 = KNearestNeighbours(random_state=7)
    clf2 = KNearestNeighbours(random_state=7)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d():
    """Tests the random initialization of K Nearest Neighbours classifier for 3d config."""

    clf1 = KNearestNeighbours(is_3d=True)
    clf2 = KNearestNeighbours(randomize=True, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier for 3d config."""

    clf1 = KNearestNeighbours(random_state=7, is_3d=True)
    clf2 = KNearestNeighbours(random_state=7, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_moon():
    """Tests the random initialization of K Nearest Neighbours classifier with moon data."""

    clf1 = KNearestNeighbours(data_shape='moon')
    clf2 = KNearestNeighbours(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_moon():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier with moon data."""

    clf1 = KNearestNeighbours(random_state=7, data_shape='moon')
    clf2 = KNearestNeighbours(random_state=7, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_moon():
    """Tests the random initialization of K Nearest Neighbours classifier for 3d config with moon data."""

    clf1 = KNearestNeighbours(is_3d=True, data_shape='moon')
    clf2 = KNearestNeighbours(randomize=True, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_moon():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier
    for 3d config with moon data."""

    clf1 = KNearestNeighbours(random_state=7, is_3d=True, data_shape='moon')
    clf2 = KNearestNeighbours(random_state=7, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_circle():
    """Tests the random initialization of K Nearest Neighbours classifier with circle data."""

    clf1 = KNearestNeighbours(data_shape='moon')
    clf2 = KNearestNeighbours(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_circle():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier with circle data."""

    clf1 = KNearestNeighbours(random_state=7, data_shape='circle')
    clf2 = KNearestNeighbours(random_state=7, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_circle():
    """Tests the random initialization of K Nearest Neighbours classifier for 3d config with circle data."""

    clf1 = KNearestNeighbours(is_3d=True, data_shape='circle')
    clf2 = KNearestNeighbours(randomize=True, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_circle():
    """Tests the initialization with a fixed random state in K Nearest Neighbours classifier
    for 3d config with circle data."""

    clf1 = KNearestNeighbours(random_state=7, is_3d=True, data_shape='circle')
    clf2 = KNearestNeighbours(random_state=7, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_decision_function():
    """Tests the decision function property for K Nearest Neighbours classifier."""

    clf = KNearestNeighbours()
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_decision_function_3d():
    """Tests the decision function property for K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(is_3d=True)
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_change_k_neighbors():
    """Tests the function to change k_neighbors in K Nearest Neighbours classifier."""

    clf = KNearestNeighbours(k_neighbors=7)
    clf.change_k_neighbors(k_neighbors=17)

    assert clf.k_neighbors == 17 and clf.classifier.n_neighbors == 17


def test_change_k_neighbors_3d():
    """Tests the function to change k_neighbors in K Nearest Neighbours classifier for 3d config."""

    clf = KNearestNeighbours(k_neighbors=7, is_3d=True)
    clf.change_k_neighbors(k_neighbors=17)

    assert clf.k_neighbors == 17 and clf.classifier.n_neighbors == 17
