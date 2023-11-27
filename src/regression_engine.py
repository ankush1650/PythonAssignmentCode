
from src.function_engine import IdealFunction


class LSRegression:
    """
    Performs least squares regression and classification of points.
    """

    def __init__(self):
        """
        Initialize LSRegression object.
        """
        pass

    def cost_reducing(self, training_function, list_of_candidate_functions, loss_function):
        """
        Reduce the cost based on the provided functions and loss function.

        Args:
            training_function (Function): Training function for comparison.
            list_of_candidate_functions (list): List of candidate functions for comparison.
            loss_function (function): Loss function for computing error.

        Returns:
            IdealFunction: Ideal function with the smallest error.
        """
        function_with_smallest_error = None
        smallest_error = None
        for function in list_of_candidate_functions:
            error = loss_function(training_function, function)
            if (smallest_error is None) or error < smallest_error:
                smallest_error = error
                function_with_smallest_error = function

        ideal_function = IdealFunction(function=function_with_smallest_error, training_function=training_function,
                                       error=smallest_error)
        return ideal_function

    def find_classification(self, point, ideal_functions):
        """
        Find the classification of a point based on the provided ideal functions.

        Args:
            point (dict): Point to be classified.
            ideal_functions (list): List of ideal functions for classification.

        Returns:
            tuple: Tuple containing the current lowest classification and its distance.
        """
        current_lowest_classification = None
        current_lowest_distance = None

        for ideal_function in ideal_functions:
            try:
                locate_y_in_classification = ideal_function.locate_y_based_on_x(point["x"])
            except IndexError:
                print("This point is not in the classification function")
                raise IndexError

            distance = abs(locate_y_in_classification - point["y"])

            if abs(distance < ideal_function.tolerance):
                if (current_lowest_classification is None) or (distance < current_lowest_distance):
                    current_lowest_classification = ideal_function
                    current_lowest_distance = distance

        return current_lowest_classification, current_lowest_distance
