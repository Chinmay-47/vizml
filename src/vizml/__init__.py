from vizml.simple_linear_regression.dashboard import simple_linear_regression_visualizer
from vizml.multi_linear_regression.dashboard import multi_linear_regression_visualizer
from vizml.k_means_clustering.dashboard import k_means_clustering_visualizer


class Visualize:

    @staticmethod
    def simple_linear_regression():
        """Runs a dashboard on localhost to visualize Simple linear regression."""
        simple_linear_regression_visualizer.run_server()

    @staticmethod
    def multi_linear_regression():
        """Runs a dashboard on localhost to visualize Multi linear regression."""
        multi_linear_regression_visualizer.run_server()

    @staticmethod
    def k_means_clustering():
        """Runs a dashboard on localhost to visualize K Means Clustering."""
        k_means_clustering_visualizer.run_server()
