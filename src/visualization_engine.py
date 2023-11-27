from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure, output_file, show

# Define color scheme and theme
color_scheme = {"train_data": "purple", "classification_function": "red", "test_point": "yellow"}
theme = "caliber"


class BokehPlot:
    """
    A class for creating Bokeh plots for visualizing data and ideal functions.
    """

    def plot_ideal_functions(self, ideal_functions, file_name):
        """
        Plot ideal functions and their corresponding training functions.

        Args:
            ideal_functions (list): List of ideal functions to plot.
            file_name (str): The name of the output file.

        Returns:
            None
        """
        # Sort ideal functions by training function name
        ideal_functions.sort(key=lambda ideal_function: ideal_function.training_function.name, reverse=False)
        plots = []
        for ideal_function in ideal_functions:
            # Plot graph for each ideal function
            p = self.plot_graph_from_two_functions(
                line_function=ideal_function,
                scatter_function=ideal_function.training_function,
                squared_error=ideal_function.error
            )
            plots.append(p)
        output_file(f"{file_name}.html")
        show(column(*plots))

    def plot_points_with_their_ideal_function(self, points_with_classification, file_name):
        """
        Plot points along with their corresponding ideal function.

        Args:
            points_with_classification (list): List of points with their classifications.
            file_name (str): The name of the output file.

        Returns:
            None
        """
        plots = []
        for index, item in enumerate(points_with_classification):
            if item["classification"] is not None:
                # Plot classification for each point
                p = self.plot_classification(item["point"], item["classification"])
                plots.append(p)
        output_file(f"{file_name}.html")
        show(column(*plots))

    def plot_graph_from_two_functions(self, scatter_function, line_function, squared_error):
        """
        Plot a graph from two functions.

        Args:
            scatter_function (function): The scatter function to plot.
            line_function (function): The line function to plot.
            squared_error (float): The squared error between the two functions.

        Returns:
            figure (obj): Bokeh figure object with the plotted graph.
        """
        f1_dataframe = scatter_function.dataframe
        f1_name = scatter_function.name

        f2_dataframe = line_function.dataframe
        f2_name = line_function.name

        squared_error = round(squared_error, 2)
        figure_two_func = figure(
            title=f"Train Model {f1_name} vs Ideal {f2_name}. Total Squared Error = {squared_error}",
            x_axis_label='X',
            y_axis_label='Y'
        )
        figure_two_func.scatter(
            f1_dataframe["x"],
            f1_dataframe["y"],
            fill_color="red",
            legend_label="Train Data",
            size=8,
            marker="circle"
        )
        figure_two_func.line(
            f2_dataframe["x"],
            f2_dataframe["y"],
            legend_label="Ideal Function",
            line_width=2,
            line_color='navy'
        )
        figure_two_func.legend.location = "bottom_right"
        figure_two_func.legend.title = "Legend"
        figure_two_func.legend.title_text_font_style = "bold"
        figure_two_func.legend.title_text_font_size = "14pt"
        figure_two_func.legend.label_text_font_size = "12pt"
        return figure_two_func

    def plot_classification(self, point, ideal_function):
        """
        Plot the classification for a point with an ideal function.

        Args:
            point (dict): The point to be plotted.
            ideal_function (function): The ideal function for classification.

        Returns:
            figure (obj): Bokeh figure object with the plotted classification.
        """
        if ideal_function is not None:
            classification_function_dataframe = ideal_function.dataframe
            x = point["x"]
            y = point["y"]
            point_str = f"({x},{y})"
            title = f"Point {point_str} with Classification: {ideal_function.name}"
            figure_classification = figure(title=title, x_axis_label='X', y_axis_label='Y')
            figure_classification.line(
                classification_function_dataframe["x"],
                classification_function_dataframe["y"],
                legend_label="Classification Function",
                line_width=2,
                line_color='navy'
            )
            criterion = ideal_function.tolerance
            classification_function_dataframe['upper'] = classification_function_dataframe['y'] + criterion
            classification_function_dataframe['lower'] = classification_function_dataframe['y'] - criterion
            source = ColumnDataSource(classification_function_dataframe.reset_index())
            band = Band(
                base='x',
                lower='lower',
                upper='upper',
                source=source,
                level='underlay',
                fill_alpha=0.3,
                line_width=1,
                line_color='red',
                fill_color="green"
            )
            figure_classification.add_layout(band)
            figure_classification.scatter(
                [point["x"]],
                [round(point["y"], 4)],
                fill_color="green",
                legend_label="Test Point",
                size=15,
                marker="triangle"
            )
            figure_classification.legend.location = "bottom_right"
            figure_classification.legend.title = "Legend"
            figure_classification.legend.title_text_font_style = "bold"
            figure_classification.legend.title_text_font_size = "14pt"
            figure_classification.legend.label_text_font_size = "12pt"
            return figure_classification
