from numpy import equal, ndarray
from plotly.graph_objects import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from vizml.bagging_classifier.classification import BaggingClassifier


def test_show_data():
    """Tests the show data function in Bagging Classifier."""

    clf = BaggingClassifier(no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary():
    """Tests the show decision boundary function in Bagging Classifier."""

    clf = BaggingClassifier(no_points=10)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities():
    """Tests the show decision probabilities function in Bagging Classifier."""

    clf = BaggingClassifier(no_points=10)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics():
    """Tests the show metrics function in Bagging Classifier."""

    clf = BaggingClassifier(no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix():
    """Tests the show confusion matrix function in Bagging Classifier."""

    clf = BaggingClassifier(no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_data_3d():
    """Tests the show data function in Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True, no_points=10)
    fig = clf.show_data(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_boundary_3d():
    """Tests the show decision boundary function in Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_decision_boundary(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_decision_probabilities_3d():
    """Tests the show decision probabilities function in Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_decision_probabilities(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_metrics_3d():
    """Tests the show metrics function in Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_metrics(return_fig=True)
    assert isinstance(fig, Figure)


def test_show_confusion_matrix_3d():
    """Tests the show silhouette plot function in Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True, no_points=10)
    clf.train()
    fig = clf.show_confusion_matrix(return_fig=True)
    assert isinstance(fig, Figure)


def test_randomize():
    """Tests the random initialization of Bagging Classifier."""

    clf1 = BaggingClassifier()
    clf2 = BaggingClassifier(randomize=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state():
    """Tests the initialization with a fixed random state in Bagging Classifier."""

    clf1 = BaggingClassifier(random_state=7)
    clf2 = BaggingClassifier(random_state=7)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d():
    """Tests the random initialization of Bagging Classifier for 3d config."""

    clf1 = BaggingClassifier(is_3d=True)
    clf2 = BaggingClassifier(randomize=True, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d():
    """Tests the initialization with a fixed random state in Bagging Classifier for 3d config."""

    clf1 = BaggingClassifier(random_state=7, is_3d=True)
    clf2 = BaggingClassifier(random_state=7, is_3d=True)
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_moon():
    """Tests the random initialization of Bagging Classifier with moon data."""

    clf1 = BaggingClassifier(data_shape='moon')
    clf2 = BaggingClassifier(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_moon():
    """Tests the initialization with a fixed random state in Bagging Classifier with moon data."""

    clf1 = BaggingClassifier(random_state=7, data_shape='moon')
    clf2 = BaggingClassifier(random_state=7, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_moon():
    """Tests the random initialization of Bagging Classifier for 3d config with moon data."""

    clf1 = BaggingClassifier(is_3d=True, data_shape='moon')
    clf2 = BaggingClassifier(randomize=True, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_moon():
    """Tests the initialization with a fixed random state in Bagging Classifier for 3d config with moon data."""

    clf1 = BaggingClassifier(random_state=7, is_3d=True, data_shape='moon')
    clf2 = BaggingClassifier(random_state=7, is_3d=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_circle():
    """Tests the random initialization of Bagging Classifier with circle data."""

    clf1 = BaggingClassifier(data_shape='moon')
    clf2 = BaggingClassifier(randomize=True, data_shape='moon')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_circle():
    """Tests the initialization with a fixed random state in Bagging Classifier with circle data."""

    clf1 = BaggingClassifier(random_state=7, data_shape='circle')
    clf2 = BaggingClassifier(random_state=7, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_randomize_3d_circle():
    """Tests the random initialization of Bagging Classifier for 3d config with circle data."""

    clf1 = BaggingClassifier(is_3d=True, data_shape='circle')
    clf2 = BaggingClassifier(randomize=True, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert not equal(data1, data2).any()


def test_random_state_3d_circle():
    """Tests the initialization with a fixed random state in Bagging Classifier for 3d config with circle data."""

    clf1 = BaggingClassifier(random_state=7, is_3d=True, data_shape='circle')
    clf2 = BaggingClassifier(random_state=7, is_3d=True, data_shape='circle')
    data1 = clf1.data_points
    data2 = clf2.data_points

    assert equal(data1, data2).all()


def test_decision_function():
    """Tests the decision function property for Bagging Classifier."""

    clf = BaggingClassifier()
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_decision_function_3d():
    """Tests the decision function property for Bagging Classifier for 3d config."""

    clf = BaggingClassifier(is_3d=True)
    clf.train()

    assert isinstance(clf.decision_function, ndarray)


def test_default_base_classifier():
    """Tests the default base classifier in Bagging Classifier."""

    clf = BaggingClassifier()

    assert isinstance(clf.classifier.base_estimator, DecisionTreeClassifier)


def test_change_base_classifier():
    """Tests the function to change the base classifier in Bagging Classifier."""

    clf = BaggingClassifier()
    clf.change_base_classifier('lr')

    assert isinstance(clf.classifier.base_estimator, LogisticRegression)


def test_change_n_estimators():
    """Tests the function to change the number of estimators in Bagging Classifier."""

    clf = BaggingClassifier()
    clf.change_n_estimators(17)

    assert clf.n_estimators == 17


def test_default_max_samples():
    """Tests the default max samples in Bagging Classifier."""

    clf = BaggingClassifier()

    assert clf.max_samples == 0.7


def test_change_max_samples():
    """Tests the function to change the max samples in Bagging Classifier."""

    clf = BaggingClassifier()
    clf.change_max_samples(0.2)

    assert clf.max_samples == 0.2
