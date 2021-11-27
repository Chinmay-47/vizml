from vizml.k_means_clustering.dashboard import DashBoard as DashBoard_KMeansClustering
from vizml.multi_linear_regression.dashboard import DashBoard as DashBoard_MultiLinearRegression
from vizml.polynomial_regression.dashboard import DashBoard as DashBoard_PolynomialRegression
from vizml.simple_linear_regression.dashboard import DashBoard as DashBoard_SimpleLinearRegression


class Visualize:

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
