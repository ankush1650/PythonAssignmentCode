import math
from src.function_engine import FunctionManager
from src.regression_engine import LSRegression
from src.visualization_engine import BokehPlot
from src.utilities.utils import write_deviation_results_to_sqlite, OLSLoss
import datetime


class RunnerAbstract:
    def __init__(self):
        self.acceptance_factor = math.sqrt(2)
        self.regression_object = LSRegression()
        self.loss_object = OLSLoss()
        self.bokeh_plot = BokehPlot()

    def run_abstract(self, ideal_dataset_path, train_dataset_path, test_dataset_path):
        candidate_ideal_manager = FunctionManager(path_of_csv=ideal_dataset_path)
        train_manager = FunctionManager(path_of_csv=train_dataset_path)

        train_manager.to_sql(file_name="output/training", suffix=" (training func)")
        candidate_ideal_manager.to_sql(file_name="output/ideal", suffix=" (ideal func)")

        ideal_functions = self._compute_ideal_functions(train_manager, candidate_ideal_manager)

        self.bokeh_plot.plot_ideal_functions(ideal_functions, "output/train_and_ideal")

        html_train_and_ideal = """Visualize the training dataset as a scatter plot, where each point represents 
        specific input-output pairs. The best fitting ideal function can be seen as a smooth curve that closely 
        follows the general trend of the data points on the scatter plot, demonstrating the expected relationship 
        between the variables. """

        print("train_ideal.html: ", html_train_and_ideal)

        test_manager = FunctionManager(path_of_csv=test_dataset_path)
        test_function = test_manager.functions[0]

        points_with_ideal_function = self._classify_test_points(test_function, ideal_functions)

        self.bokeh_plot.plot_points_with_their_ideal_function(points_with_ideal_function, "output/point_and_ideal")

        write_deviation_results_to_sqlite(points_with_ideal_function)

        print("points_and_ideal.html: Display a visual representation of the points that share a matching ideal "
              "function, illustrating the distances between them within a figure.")
        self._display_results()

    def _compute_ideal_functions(self, train_manager, candidate_ideal_manager):
        ideal_functions = []
        for train_func in train_manager:
            ideal_function = self.regression_object.cost_reducing(training_function=train_func,
                                                                  list_of_candidate_functions=candidate_ideal_manager.functions,
                                                                  loss_function=self.loss_object.ordinary_squared_error)
            ideal_function.tolerance_factor = self.acceptance_factor
            ideal_functions.append(ideal_function)
        return ideal_functions

    def _classify_test_points(self, test_func, ideal_functions):
        points_with_ideal_function = []
        for point in test_func:
            ideal_function, delta_y = self.regression_object.find_classification(point=point, ideal_functions=ideal_functions)
            result = {"point": point, "classification": ideal_function, "delta_y": delta_y}
            points_with_ideal_function.append(result)
        return points_with_ideal_function

    def _display_results(self):
        print("Check output directory")
        print("Date of execution:", datetime.datetime.now())
        print("Program executed!!")


if __name__ == '__main__':
    runner_abstract = RunnerAbstract()
    ideal_dataset_path = "dataset/ideal.csv"
    train_dataset_path = "dataset/train.csv"
    test_dataset_path = "dataset/test.csv"
    runner_abstract.run_abstract(ideal_dataset_path, train_dataset_path, test_dataset_path)
