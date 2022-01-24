from numpy import equal, ndarray
from plotly.graph_objects import Figure
from vizml.support_vector_machine.classification import SupportVectorMachine


def test_show_data():
    """Tests the show data function in Support Vector Machine."""

    clf = SupportVectorMachine(no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary():
    """Tests the show decision boundary function in Support Vector Machine."""

    clf = SupportVectorMachine(no_points=10)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities():
    """Tests the show decision probabilities function in Support Vector Machine."""

    clf = SupportVectorMachine(no_points=10)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics():
    """Tests the show metrics function in Support Vector Machine."""

    clf = SupportVectorMachine(no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix():
    """Tests the show confusion matrix function in Support Vector Machine."""

    clf = SupportVectorMachine(no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_3d():
    """Tests the show data function in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True, no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary_3d():
    """Tests the show decision boundary function in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities_3d():
    """Tests the show decision probabilities function in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics_3d():
    """Tests the show metrics function in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix_3d():
    """Tests the show silhouette plot function in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize():
    """Tests the random initialization of Support Vector Machine."""

    clf1 = SupportVectorMachine()
    clf2 = SupportVectorMachine(randomize=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in Support Vector Machine."""

    clf1 = SupportVectorMachine(random_state=7)
    clf2 = SupportVectorMachine(random_state=7)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d():
    """Tests the random initialization of Support Vector Machine for 3d config."""

    clf1 = SupportVectorMachine(is_3d=True)
    clf2 = SupportVectorMachine(randomize=True, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d():
    """Tests the initialization with a fixed random state in Support Vector Machine for 3d config."""

    clf1 = SupportVectorMachine(random_state=7, is_3d=True)
    clf2 = SupportVectorMachine(random_state=7, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_moon():
    """Tests the random initialization of Support Vector Machine with moon data."""

    clf1 = SupportVectorMachine(data_shape='moon')
    clf2 = SupportVectorMachine(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_moon():
    """Tests the initialization with a fixed random state in Support Vector Machine with moon data."""

    clf1 = SupportVectorMachine(random_state=7, data_shape='moon')
    clf2 = SupportVectorMachine(random_state=7, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_moon():
    """Tests the random initialization of Support Vector Machine for 3d config with moon data."""

    clf1 = SupportVectorMachine(is_3d=True, data_shape='moon')
    clf2 = SupportVectorMachine(randomize=True, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_moon():
    """Tests the initialization with a fixed random state in Support Vector Machine for 3d config with moon data."""

    clf1 = SupportVectorMachine(random_state=7, is_3d=True, data_shape='moon')
    clf2 = SupportVectorMachine(random_state=7, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_circle():
    """Tests the random initialization of Support Vector Machine with circle data."""

    clf1 = SupportVectorMachine(data_shape='moon')
    clf2 = SupportVectorMachine(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_circle():
    """Tests the initialization with a fixed random state in Support Vector Machine with circle data."""

    clf1 = SupportVectorMachine(random_state=7, data_shape='circle')
    clf2 = SupportVectorMachine(random_state=7, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_circle():
    """Tests the random initialization of Support Vector Machine for 3d config with circle data."""

    clf1 = SupportVectorMachine(is_3d=True, data_shape='circle')
    clf2 = SupportVectorMachine(randomize=True, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_circle():
    """Tests the initialization with a fixed random state in Support Vector Machine for 3d config with circle data."""

    clf1 = SupportVectorMachine(random_state=7, is_3d=True, data_shape='circle')
    clf2 = SupportVectorMachine(random_state=7, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_decision_function():
    """Tests the decision function property for Support Vector Machine."""

    clf = SupportVectorMachine()
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_decision_function_3d():
    """Tests the decision function property for Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True)
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_support_vectors():
    """Tests the support vectors property for Support Vector Machine."""

    clf = SupportVectorMachine()
    clf.train()

    assert isinstance(clf.support_vectors, ndarray)


def test_support_vectors_3d():
    """Tests the support vectors property for Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(is_3d=True)
    clf.train()

    assert isinstance(clf.support_vectors, ndarray)


def test_change_kernel():
    """Tests the function to change the kernel in Support Vector Machine."""

    clf = SupportVectorMachine(kernel='poly')
    clf.change_kernel('rbf')

    assert clf.kernel == 'rbf' and clf.classifier.kernel == 'rbf'


def test_change_kernel_3d():
    """Tests the function to change the kernel in Support Vector Machine for 3d config."""

    clf = SupportVectorMachine(kernel='poly', is_3d=True)
    clf.change_kernel('rbf')

    assert clf.kernel == 'rbf' and clf.classifier.kernel == 'rbf'
