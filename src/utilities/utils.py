import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Band
from bokeh.plotting import figure
from sqlalchemy import create_engine, MetaData, Column, Table, Float, String
from sqlalchemy.exc import IntegrityError


class FunctionManager:
    """
     Manages a set of functions and provides methods for handling them.
     """

    def __init__(self, path_of_csv):
        self._functions = []
        try:
            self._function_data = pd.read_csv(path_of_csv, index_col=False)
        except FileNotFoundError:
            print("Issue while reading file {}".format(path_of_csv))
            raise

        x_values = self._function_data["x"]
        for name_of_column, data_of_column in self._function_data.items():
            if "x" in name_of_column:
                continue
            subset = pd.concat([x_values, data_of_column], axis=1)
            function = Function.from_dataframe(name_of_column, subset)
            self._functions.append(function)

    def to_sql(self, file_name, suffix):
        """
              Write function data to a SQLite database.

              Args:
                  file_name (str): Name of the output file.
                  suffix (str): Suffix to be added to column names.

              Returns:
                  None
              """
        engine = create_engine(f'sqlite:///{file_name}.db', echo=False)
        copy_of_function_data = self._function_data.copy()
        copy_of_function_data.columns = [name.capitalize() + suffix for name in copy_of_function_data.columns]
        copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

        copy_of_function_data.to_sql(
            file_name,
            engine,
            if_exists="replace",
            index=True,
        )

    @property
    def functions(self):
        return self._functions

    def __iter__(self):
        return FunctionManagerIterator(self)

    def __repr__(self):
        return "Contains {} number of functions".format(len(self.functions))


class FunctionManagerIterator():
    def __init__(self, function_manager):
        self._index = 0
        self._function_manager = function_manager

    def __next__(self):
        if self._index < len(self._function_manager.functions):
            value_requested = self._function_manager.functions[self._index]
            self._index = self._index + 1
            return value_requested
        raise StopIteration


class Function:
    def __init__(self, name):
        self._name = name
        self.dataframe = pd.DataFrame()

    def locate_y_based_on_x(self, x):
        search_key = self.dataframe["x"] == x
        try:
            return self.dataframe.loc[search_key].iat[0, 1]
        except IndexError:
            raise IndexError

    @property
    def name(self):
        return self._name

    def __iter__(self):
        return FunctionIterator(self)

    def __sub__(self, other):
        diff = self.dataframe - other.dataframe
        return diff

    @classmethod
    def from_dataframe(cls, name, dataframe):
        function = cls(name)
        function.dataframe = dataframe
        function.dataframe.columns = ["x", "y"]
        return function

    def __repr__(self):
        return "Function for {}".format(self.name)


class IdealFunction(Function):
    def __init__(self, function, training_function, error):
        super().__init__(function.name)
        self.dataframe = function.dataframe

        self.training_function = training_function
        self.error = error
        self._tolerance_value = 1
        self._tolerance = 1

    def _determine_largest_deviation(self, ideal_function, train_function):
        distances = train_function - ideal_function
        distances["y"] = distances["y"].abs()
        largest_deviation = max(distances["y"])
        return largest_deviation

    @property
    def tolerance(self):
        self._tolerance = self.tolerance_factor * self.largest_deviation
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def tolerance_factor(self):
        return self._tolerance_value

    @tolerance_factor.setter
    def tolerance_factor(self, value):
        self._tolerance_value = value

    @property
    def largest_deviation(self):
        largest_deviation = self._determine_largest_deviation(self, self.training_function)
        return largest_deviation


class FunctionIterator:
    def __init__(self, function):
        self._function = function
        self._index = 0

    def __next__(self):
        if self._index < len(self._function.dataframe):
            value_requested_series = (self._function.dataframe.iloc[self._index])
            point = {"x": value_requested_series.x, "y": value_requested_series.y}
            self._index += 1
            return point
        raise StopIteration


def squared_error(first_function, second_function):
    distances = second_function - first_function
    distances["y"] = distances["y"] ** 2
    total_deviation = sum(distances["y"])
    return total_deviation


def plot_ideal_functions(ideal_functions, file_name):
    ideal_functions.sort(key=lambda ideal_function: ideal_function.training_function.name, reverse=False)
    plots = []
    for ideal_function in ideal_functions:
        p = plot_graph_from_two_functions(line_function=ideal_function,
                                          scatter_function=ideal_function.training_function,
                                          squared_error=ideal_function.error)
        plots.append(p)
    output_file("{}.html".format(file_name))
    show(column(*plots))


def plot_points_with_their_ideal_function(points_with_classification, file_name):
    plots = []
    for index, item in enumerate(points_with_classification):
        if item["classification"] is not None:
            p = plot_classification(item["point"], item["classification"])
            plots.append(p)
    output_file("{}.html".format(file_name))
    show(column(*plots))


def plot_graph_from_two_functions(scatter_function, line_function, squared_error):
    f1_dataframe = scatter_function.dataframe
    f1_name = scatter_function.name

    f2_dataframe = line_function.dataframe
    f2_name = line_function.name

    squared_error = round(squared_error, 2)
    p = figure(title="train model {} vs ideal {}. Total squared error = {}".format(f1_name, f2_name, squared_error),
               x_axis_label='x', y_axis_label='y')
    p.scatter(f1_dataframe["x"], f1_dataframe["y"], fill_color="red", legend_label="Train")
    p.line(f2_dataframe["x"], f2_dataframe["y"], legend_label="Ideal", line_width=2)
    return p


def plot_classification(point, ideal_function):
    if ideal_function is not None:
        classification_function_dataframe = ideal_function.dataframe

        point_str = "({},{})".format(point["x"], point["y"])
        title = f"point {point_str} with classification: {ideal_function.name}"

        p = figure(title=title, x_axis_label='x', y_axis_label='y')

        p.line(classification_function_dataframe["x"], classification_function_dataframe["y"],
               legend_label="Classification function", line_width=2, line_color='black')

        criterion = ideal_function.tolerance
        classification_function_dataframe['upper'] = classification_function_dataframe['y'] + criterion
        classification_function_dataframe['lower'] = classification_function_dataframe['y'] - criterion

        source = ColumnDataSource(classification_function_dataframe.reset_index())

        band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay',
                    fill_alpha=0.3, line_width=1, line_color='green', fill_color="green")

        p.add_layout(band)

        p.scatter([point["x"]], [round(point["y"], 4)], fill_color="red", legend_label="Test point", size=8)

        return p


def minimise_loss(training_function, list_of_candidate_functions, loss_function):
    function_with_smallest_error = None
    smallest_error = None
    for function in list_of_candidate_functions:
        error = loss_function(training_function, function)
        if ((smallest_error == None) or error < smallest_error):
            smallest_error = error
            function_with_smallest_error = function

    ideal_function = IdealFunction(function=function_with_smallest_error, training_function=training_function,
                                   error=smallest_error)
    return ideal_function


def find_classification(point, ideal_functions):
    current_lowest_classification = None
    current_lowest_distance = None

    for ideal_function in ideal_functions:
        try:
            locate_y_in_classification = ideal_function.locate_y_based_on_x(point["x"])
        except IndexError:
            print("This point is not in the classification function")
            raise IndexError

        distance = abs(locate_y_in_classification - point["y"])

        if (abs(distance < ideal_function.tolerance)):
            if ((current_lowest_classification == None) or (distance < current_lowest_distance)):
                current_lowest_classification = ideal_function
                current_lowest_distance = distance

    return current_lowest_classification, current_lowest_distance


def write_deviation_results_to_sqlite(result):
    output_path = "output/mapping"
    DB_URL = f'sqlite:///{output_path}.db'
    engine = create_engine(DB_URL, echo=False)
    metadata = MetaData()

    mapping = Table('mapping', metadata,
                    Column('X (test func)', Float, primary_key=True),
                    Column('Y (test func)', Float),
                    Column('Delta Y (test func)', Float),
                    Column('No. of ideal func', String(50))
                    )

    metadata.create_all(engine)

    execute_map = []
    for item in result:
        point = item["point"]
        classification = item["classification"]
        delta_y = item["delta_y"]

        if classification is not None:
            classification_name = classification.name
        else:
            classification_name = "-"
            delta_y = -1

        execute_map.append(
            {"X (test func)": point["x"], "Y (test func)": point["y"], "Delta Y (test func)": delta_y,
             "No. of ideal func": classification_name})

        with engine.connect() as connection:
            trans = connection.begin()
            try:
                for entry in execute_map:
                    try:
                        connection.execute(mapping.insert().values(entry))
                    except IntegrityError:
                        # Handle the case when the entry already exists
                        existing_entry = connection.execute(
                            mapping.select().where(mapping.c['X (test func)'] == entry["X (test func)"])).fetchone()
                        if existing_entry is not None:
                            # Perform an update on the existing entry
                            connection.execute(
                                mapping.update().where(mapping.c['X (test func)'] == entry["X (test func)"]).values(
                                    {
                                        "Y (test func)": entry["Y (test func)"],
                                        "Delta Y (test func)": entry["Delta Y (test func)"],
                                        "No. of ideal func": entry["No. of ideal func"]
                                    }
                                )
                            )
                        else:
                            # Log the error for any other unexpected cases
                            print("Unexpected IntegrityError occurred.")
                            raise
                trans.commit()
            except Exception as e:
                # Rollback the transaction
                trans.rollback()
                print(f"Error occurred: {e}")
                raise


class OLSLoss:
    """
    Provides methods for calculating ordinary least squares (OLS) loss.
    """

    def ordinary_squared_error(self, first_function, second_function):
        """
        Calculate the ordinary squared error between two functions.

        Args:
            first_function (pandas.DataFrame): First function for comparison.
            second_function (pandas.DataFrame): Second function for comparison.

        Returns:
            total_deviation (float): Total deviation between the two functions.
        """
        distances = second_function - first_function
        if distances.empty:  # You should check if the DataFrame is empty.
            return 0
        distances["y"] = distances["y"] ** 2
        total_deviation = sum(distances["y"])
        if not self.check_if_number(total_deviation):
            return 0  # Return a value in case the check fails.
        return total_deviation

    def check_if_number(self, variable):
        """
        Check if a variable is a number.

        Args:
            variable (int or float): Variable to check.

        Returns:
            bool: True if the variable is a number, otherwise raises an error.
        """
        if not isinstance(variable, (int, float)):
            raise ValueError("Variable is not a number.")
        return True
