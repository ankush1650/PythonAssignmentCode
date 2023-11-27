import pandas as pd
from sqlalchemy import create_engine


class FunctionManager:
    def __init__(self, path_of_csv):
        """
                Initialize FunctionManager with data from a CSV file.

                Args:
                    path_of_csv (str): Path to the CSV file.

                Raises:
                    FileNotFoundError: If the specified file is not found.
                """
        self._functions = []
        try:
            self._function_data = pd.read_csv(path_of_csv, index_col=False)
        except FileNotFoundError:
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


class FunctionManagerIterator:
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
