from vizml.adaboost_classifier.dashboard import DashBoard as DashBoard_AdaBoostClassifier
from vizml.bagging_classifier.dashboard import DashBoard as DashBoard_BaggingClassifier
from vizml.dbscan.dashboard import DashBoard as DashBoard_DBScan
from vizml.decision_tree.dashboard import DashBoard as DashBoard_DecisionTree
from vizml.k_means_clustering.dashboard import DashBoard as DashBoard_KMeansClustering
from vizml.k_nearest_neighbours.dashboard import DashBoard as DashBoard_KNN
from vizml.logistic_regression.dashboard import DashBoard as DashBoard_LogisticRegression
from vizml.multi_linear_regression.dashboard import DashBoard as DashBoard_MultiLinearRegression
from vizml.naive_bayes.dashboard import DashBoard as DashBoard_NaiveBayes
from vizml.polynomial_regression.dashboard import DashBoard as DashBoard_PolynomialRegression
from vizml.simple_linear_regression.dashboard import DashBoard as DashBoard_SimpleLinearRegression
from vizml.support_vector_machine.dashboard import DashBoard as DashBoard_SupportVectorMachines


class Visualize:
    """Aggregator class to run visualizations."""

    @staticmethod
    def simple_linear_regression():
        """Runs a dashboard on localhost to visualize Simple linear regression."""
        DashBoard_SimpleLinearRegression().run()

    @staticmethod
    def multi_linear_regression():
        """Runs a dashboard on localhost to visualize Multi linear regression."""
        DashBoard_MultiLinearRegression().run()

    @staticmethod
    def k_means_clustering():
        """Runs a dashboard on localhost to visualize K Means Clustering."""
        DashBoard_KMeansClustering().run()

    @staticmethod
    def polynomial_regression():
        """Runs a dashboard on localhost to visualize Polynomial Regression."""
        DashBoard_PolynomialRegression().run()

    @staticmethod
    def dbscan():
        """Runs a dashboard on localhost to visualize DBScan."""
        DashBoard_DBScan().run()

    @staticmethod
    def logistic_regression():
        """Runs a dashboard on localhost to visualize Logistic Regression."""
        DashBoard_LogisticRegression().run()

    @staticmethod
    def support_vector_machines():
        """Runs a dashboard on localhost to visualize Support Vector Machines."""
        DashBoard_SupportVectorMachines().run()

    @staticmethod
    def k_nearest_neighbors():
        """Runs a dashboard on localhost to visualize K Nearest Neighbors Classifier."""
        DashBoard_KNN().run()

    @staticmethod
    def naive_bayes():
        """Runs a dashboard on localhost to visualize Naive Bayes Classifier."""
        DashBoard_NaiveBayes().run()

    @staticmethod
    def decision_tree():
        """Runs a dashboard on localhost to visualize Decision Tree Classifier."""
        DashBoard_DecisionTree().run()

    @staticmethod
    def bagging_classifier():
        """Runs a dashboard on localhost to visualize Bagging Classifier."""
        DashBoard_BaggingClassifier().run()

    @staticmethod
    def adaboost_classifier():
        """Runs a dashboard on localhost to visualize AdaBoost Classifier."""
        DashBoard_AdaBoostClassifier().run()
