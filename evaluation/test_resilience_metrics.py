import numpy as np
import unittest
import plotly.graph_objects as go
from resmetric.plot import create_plot_from_data

# Boolean flag to control whether to display test cases plot before testing with the module or not
SHOW_TEST_CASES = False

# Boolean flag to control whether to display the plot after testing with the module or not
SHOW_PLOT = True

# Define the range of x-values for the plot (0 to RANGE)
RANGE = 20


# Here, you can define your own test-cases
# First, for x in [0,x0], There will be a plateau of y0
# x in [x0, x1]: - a parabola will be fitted to have its minimum y_min=k (and h is neglected)
#                - or a triangle where the middle point is at (h,k)
# x in [x1, RANGE]: 2nd plateau of y_1
VALUES = {
    'root': (4, .95, 16, .85, 7, .6),  # (x0, y0, x1, y1, h, k)
    'shallow': (4, .95, 16, .95, 9, .8),
    'deep': (4, .95, 16, .9, 7, .3),
    'long': (2, .9, 19, .75, 12, .6)
}


# Helper function to create a synthetic dip in y-values
def generate_parabolical_dip(test_case: str):
    """
    This function generates a parabolical dip of a given test case.
    For this, it fits a parabola through (x0, y0) and (x1, y1) with the vertex's y-value of k.
    If you have some time and want to improve your knowledge of parabola-fitting, you can work through the code.
    If not, just accept it "as-is" ;)

    Args:
        test_case: One of the test cases as defined above in VALUES

    Returns:
        Figure as defined above in comment of VALUES

    """
    def calculate_parameters(x1, y1, x2, y2, k):
        # Calculate the square roots of the differences
        sqrt_y1_k = np.sqrt(np.abs(y1 - k))
        sqrt_y2_k = np.sqrt(np.abs(y2 - k))

        # Calculate the parameters t for both cases
        t1 = sqrt_y1_k / (sqrt_y1_k + sqrt_y2_k)
        t2 = sqrt_y2_k / (sqrt_y1_k + sqrt_y2_k)

        # Calculate h for both cases
        h1 = (1 - t1) * x1 + t1 * x2
        h2 = (1 - t2) * x1 + t2 * x2

        return t1, h1, t2, h2

    def calculate_parabola_coefficients(x1, y1, x2, y2, k, h):
        if (x1 - h) ** 2 != (x2 - h) ** 2:  # Make sure points are not symmetrical around the vertex
            a = (y1 - y2) / ((x1 - h) ** 2 - (x2 - h) ** 2)
            b = (y1 - a * (x1 - h) ** 2 - k) / (x1 - h)
        else:
            # If symmetrical around h, treat the problem carefully as b becomes 0
            a = (y1 - k) / (x1 - h) ** 2
            b = 0  # Symmetrical, so no linear term
        return a, b

    def check_parabola(x1, y1, x2, y2, k, a, b, h):
        # Check if the parabola passes through all three points
        y1_check = np.isclose(a * (x1 - h) ** 2 + b * (x1 - h) + k, y1, atol=1e-6)
        y2_check = np.isclose(a * (x2 - h) ** 2 + b * (x2 - h) + k, y2, atol=1e-6)
        # Ensure the vertex h is between x1 and x2
        y_min_check = (x1 <= h and h <= x2) or (x2 <= h and h <= x1)

        return y1_check and y2_check and y_min_check

    # Generate two possible parabolas that fit the given points and vertex
    def generate_parabolas(x1, y1, x2, y2, k):
        # Calculate parameters t and h for both parabolas
        t1, h1, t2, h2 = calculate_parameters(x1, y1, x2, y2, k)

        # Calculate coefficients a and b for both parabolas
        a1, b1 = calculate_parabola_coefficients(x1, y1, x2, y2, k, h1)
        a2, b2 = calculate_parabola_coefficients(x1, y1, x2, y2, k, h2)

        # Generate x values over the specified range
        x_values = np.arange(RANGE + 1)

        # Calculate y values for both parabolas
        y1_values = a1 * (x_values - h1) ** 2 + b1 * (x_values - h1) + k
        y2_values = a2 * (x_values - h2) ** 2 + b2 * (x_values - h2) + k

        return x_values, y1_values, y2_values, a1, b1, h1, a2, b2, h2

    assert test_case in VALUES.keys(), f"Test Case '{test_case}' is not defined!"
    # Given values
    x0, y0, x1, y1, _, k = VALUES.get(test_case)

    # Generate the y values for both parabolas
    x_values, y1_values, y2_values, a1, b1, h1, a2, b2, h2 = generate_parabolas(x0, y0, x1, y1, k)

    # Check which parabola fits all three points
    fits_parabola1 = check_parabola(x0, y0, x1, y1, k, a1, b1, h1)
    fits_parabola2 = check_parabola(x0, y0, x1, y1, k, a2, b2, h2)

    # Assertions to check if the parabolas correctly fit the points
    assert fits_parabola1 or fits_parabola2, "Test-case construction failed - Neither parabola fits all three points."

    # Initialize y values with y0
    y_values = np.ones(RANGE + 1) * y0

    # Apply the parabola between x1 and x2
    a = a1 if fits_parabola1 else a2
    h = h1 if fits_parabola1 else h2
    mask_parabola = (x_values >= x0) & (x_values <= x1)
    y_values[mask_parabola] = a * (x_values[mask_parabola] - h) ** 2 + k

    # Set y2 for x >= x2
    mask_y1 = (x_values > x1)
    y_values[mask_y1] = y1

    # Create the plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=test_case, line=dict(color='blue')))

    return fig


def generate_triangle_dip(test_case: str):
    assert test_case in VALUES.keys(), f"Test Case '{test_case}' is not defined!"
    # Given values
    x0, y0, x1, y1, x_k, k = VALUES.get(test_case)

    # Initialize y values with y0
    x_values = np.arange(RANGE + 1)
    y_values = np.ones(RANGE + 1) * y0

    # Linear descent from (x0, y0) to (x_k, k)
    mask_descent = (x_values >= x0) & (x_values <= x_k)
    y_values[mask_descent] = y0 + (k - y0) * (x_values[mask_descent] - x0) / (x_k - x0)

    # Linear ascent from (x_k, k) to (x1, y1)
    mask_ascent = (x_values > x_k) & (x_values <= x1)
    y_values[mask_ascent] = k + (y1 - k) * (x_values[mask_ascent] - x_k) / (x1 - x_k)

    # Set y_values for x > x1 to y1
    mask_y1 = (x_values > x1)
    y_values[mask_y1] = y1

    # Create the plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=test_case, line=dict(color='blue')))

    return fig


class TestResilienceMetricsParabolic(unittest.TestCase):
    def test_parabolic_root(self):
        test_case_name = "root"

        # Create synthetic y-values with a dip
        fig = generate_parabolical_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_parabolic_shallow(self):
        test_case_name = "shallow"

        # Create synthetic y-values with a dip
        fig = generate_parabolical_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_parabolic_deep(self):
        test_case_name = "deep"

        # Create synthetic y-values with a dip
        fig = generate_parabolical_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_parabolic_long(self):
        test_case_name = "long"

        # Create synthetic y-values with a dip
        fig = generate_parabolical_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")


class TestResilienceMetricsTriangular(unittest.TestCase):
    def test_triangular_root(self):
        test_case_name = "root"

        # Create synthetic y-values with a dip
        fig = generate_triangle_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_triangular_shallow(self):
        test_case_name = "shallow"

        # Create synthetic y-values with a dip
        fig = generate_triangle_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_triangular_deep(self):
        test_case_name = "deep"

        # Create synthetic y-values with a dip
        fig = generate_triangle_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")

    def test_triangular_long(self):
        test_case_name = "long"

        # Create synthetic y-values with a dip
        fig = generate_triangle_dip(test_case_name)

        if SHOW_TEST_CASES:
            fig.show()

        # Get the JSON representation of the figure
        json_data = fig.to_json()

        # Step 2: Generate the plot with additional traces using the JSON data
        fig = create_plot_from_data(
            json_data,
            include_count_below_thresh=True,
            include_maximal_dips=True,
            include_bars=True
        )

        if SHOW_PLOT:
            fig.show()

        # Assert that figure is not empty
        self.assertGreater(len(fig.data), 0, "Figure data should not be empty")

        # Assert that more than one trace is added
        self.assertGreater(len(fig.data), 1, "Figure should have more than one trace after processing")


if __name__ == '__main__':
    unittest.main()
