import numpy as np
import math
from scipy.signal import find_peaks

import pwlf
from skopt import gp_minimize
from skopt.space import Integer


def calculate_auc(y_values):
    """
    Calculate the Area Under the Curve (AUC) using a cumulative average approach.

    This function computes the AUC by calculating the cumulative average of the
    input values at each point. The AUC is a measure of the aggregate value of
    the data points, providing a way to evaluate the overall level of the input
    series.

    Parameters:
    - y_values (array-like): Input data array. Should be a list or array of numerical values.

    Returns:
    - list: A list of AUC values at each point in the input data. The length of this
            list will be the same as the input `y_values`, where each entry corresponds
            to the AUC calculated up to that point.

    Example:
    >>> y_values = [1, 2, 3, 4]
    >>> calculate_auc(y_values)
    [1.0, 1.5, 2.0, 2.5]

    Notes:
    - The function computes the AUC incrementally, adding the current value to a cumulative
      sum and then dividing by the current index plus one. This provides a running average
      of the AUC values.
    """
    auc_values = []
    cumulative_sum = 0
    for i, y in enumerate(y_values):
        cumulative_sum += y
        auc = cumulative_sum / (i + 1)
        auc_values.append(auc)
    return auc_values


def calculate_inverse_weighted_auc(y_values):
    """
    Calculate the AUC with inverse weighting.

    Parameters:
    - y_values: List of y-values for which the AUC is calculated.

    Returns:
    - auc_values: List of inverse weighted AUC values.
    """
    auc_values = []
    cumulative_sum = 0
    weights_sum = 0

    for i, y in enumerate(y_values):
        # Calculate weight based on inverse function
        index = len(y_values) - i
        weight = 1 / (index + 1)  # Note: i + 1 to avoid division by zero

        # Update cumulative sum with weighted value
        cumulative_sum += y * weight

        # Update the sum of weights
        weights_sum += weight

        # Calculate the weighted AUC up to the current point
        auc = cumulative_sum / weights_sum
        auc_values.append(auc)

    return auc_values


def calculate_weighted_auc(y_values, lambda_weight):
    """
    Calculate the weighted AUC with exponential decay weighting.

    Parameters:
    - y_values: List of y-values for which the AUC is calculated.
    - lambda_weight: Weighting factor for the exponential decay function.

    Returns:
    - auc_values: List of weighted AUC values.
    """
    auc_values = []
    cumulative_sum = 0
    weights_sum = 0

    for i, y in enumerate(y_values):
        # Calculate weight based on exponential decay
        index = len(y_values) - i
        weight = np.exp(-lambda_weight * index)

        # Update cumulative sum with weighted value
        cumulative_sum += y * weight

        # Update the sum of weights
        weights_sum += weight

        # Calculate the weighted AUC up to the current point
        auc = cumulative_sum / weights_sum
        auc_values.append(auc)

    return auc_values


def _make_color_pale_hex(hex_color, factor=0.5):
    """
    Make a color more pale by blending it with white.

    :param hex_color: Color in HTML format (e.g., "#RRGGBB").
    :param factor: Blending factor where 0 is the original color and 1 is pure white.
    :return: Pale color in HTML format.
    """
    if not (0 <= factor <= 1):
        raise ValueError("Factor must be between 0 and 1.")

    # Remove the '#' and ensure it has 6 characters
    hex_color = hex_color.lstrip('#')
    hex_color = hex_color.zfill(6)

    # Extract RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Blend with white
    r_pale = int(r + (255 - r) * factor)
    g_pale = int(g + (255 - g) * factor)
    b_pale = int(b + (255 - b) * factor)

    # Convert back to hex and return
    return "#{:02X}{:02X}{:02X}".format(r_pale, g_pale, b_pale)


def _half_life_to_lambda(half_life):
    """
    Convert half-life to decay parameter lambda.

    Parameters:
    half_life (float): The half-life in units.

    Returns:
    float: The decay parameter lambda.
    """
    return math.log(2) / half_life


def time_below_threshold(y_values, threshold):
    """
    Count how many times values fall below a given threshold.

    Parameters:
    - y_values: List of y-values to be analyzed.
    - threshold: Value below which occurrences are counted.

    Returns:
    - count_list: List of counts of values below the threshold for each point.
    """
    # Generate count list with zeros
    count_list = [np.sum(np.array(y_values[:i + 1]) < threshold / 100) for i in range(len(y_values))]
    return count_list


def count_dibs_below_threshold_series(y_values, threshold):
    """
    Count how many times values fall below a given threshold.

    Parameters:
    - y_values: List of y-values to be analyzed.
    - threshold: Value below which occurrences are counted in percent, i.e., 80 for 80%

    Returns:
    - count_list: List of counts of values below the threshold for each point.
    """
    # Generate list that masks the times, when the time series is below the given threshold
    below_list = (np.array(y_values) < threshold / 100).astype(int)

    def _sign_changes_series(series):
        # sign changes in a binary series
        sign_changes_list = [0]  # no sign change at index 0 possible
        for i in range(len(series)):
            if i == 0:  # no sign change at index 0 possible
                continue
            sign_changes_list.append(series[i] != series[i - 1])
        return sign_changes_list

    # Generate the count_list by iterating over each index in below_list
    count_list = [
        math.ceil(  # Apply the ceiling function to round up to the nearest integer. If only crossed once => also dip
            (
                # Sum the sign changes up to the current index (i + 1)
                    np.sum(_sign_changes_series(below_list[:i + 1]))
                    + (1 if below_list[0] == 1 else 0)  # Account for the case when time series starts below
                # threshold. This would account for the first dip in this case.
            )
            / 2  # (The boundary must be crossed twice for one dip)
        )
        for i in range(len(below_list))  # Loop through each index in below_list
    ]
    return count_list


def calculate_max_drawdown(time_series):
    """
    Calculate the maximum drawdown of a time series.

    Parameters:
    ----------
    time_series : list of float
        A list of numeric values representing the time series.

    Returns:
    -------
    max_drawdown : float
        The maximum drawdown as a percentage.

    drawdown_series : list of float
        A list representing the relative loss at each point in the time series,
        calculated as the difference between the current value and the highest
        value reached up to that point, divided by that highest value.
    """

    # Convert the list to a numpy array for easier manipulation
    time_series = np.array(time_series)

    # Calculate the cumulative maximum of the time series
    cumulative_max = np.maximum.accumulate(time_series)

    # Calculate the drawdown as the difference between the current value and the cumulative max
    drawdown = (time_series - cumulative_max) / cumulative_max

    # Find the maximum drawdown (the minimum value in the drawdown series)
    max_drawdown = drawdown.min()

    # Convert the drawdown series back to a list for consistency
    drawdown_series = drawdown.tolist()

    return max_drawdown, drawdown_series


def detect_peaks(y_values):
    """
        Detect and return peaks in a time series.

        Parameters:
        - y_values: List-like object of y-values for peak detection.

        Returns:
        - peaks: Indices of detected peaks.
    """
    if not len(y_values) > 1:
        raise ValueError("y_values must be at least two values")

    # Add padding with -1 at the beginning and end of y_values
    padded_y_values = np.concatenate(([-1], y_values, [-1]))

    peaks, _ = find_peaks(padded_y_values)  #, prominence=prominence, width=width)

    # Adjust indices to correspond to the original y_values
    peaks -= 1  # Subtract 1 to correct for the padding

    return peaks


def _find_next_smaller(values, target):
    """
    Find the largest value in the ordered list that is smaller than the target value using binary search.

    Parameters:
    - values: Ordered list of values to search in.
    - target: The target value to compare against.

    Returns:
    - The largest value that is smaller than the target, or None if no such value exists.
    """
    left, right = 0, len(values) - 1
    result = None

    while left <= right:
        mid = (left + right) // 2
        if values[mid] < target:
            result = values[mid]  # This could be a potential answer
            left = mid + 1  # Look for a potentially larger smaller value
        else:
            right = mid - 1  # Look for smaller values

    return result


def _get_dips(values, maxs=None):
    """
    Identify dips in a time series based on detected peaks (maxima).

    Parameters:
    - values: List or array of numerical values representing the time series data.
    - maxs: List of indices (timestamps) corresponding to the detected peaks (maxima)
            in the time series. If not provided, peaks will be detected automatically.

    Returns:
    - dips: List of tuples where each tuple represents the start and end indices
            (timestamps) of a dip.
    """

    if maxs is None:
        maxs = detect_peaks(np.array(values))  # Detect peaks if not provided

    dips = []  # Initialize a list to store the dips

    for j, maxi in enumerate(maxs):
        """
        maxi: An individual index (timestamp) corresponding to a detected peak (maximum) 
              in the time series.
        """
        prev_max_height = 0  # Track the previous maximum height within the current dip

        # Filter the maxs list to only include peaks that occur after the current peak (maxi)
        subsequent_maxs = [m for m in maxs if m > maxi]
        """
        subsequent_maxs: A list of indices (timestamps) that correspond to peaks (maxima)
                         occurring after the current peak (maxi).
        """

        # Iterate over these subsequent peaks
        for max_next in subsequent_maxs:
            """
            max_next: An individual index (timestamp) corresponding to a subsequent 
                      detected peak (maximum) that occurs after the current peak (maxi).
            """

            # Check if the value at the next peak is lower than the previous maximum height
            if values[max_next] < prev_max_height:
                continue

            prev_max_height = values[max_next]  # Update the previous max height
            dips.append((maxi, max_next))  # Record the dip

            # Stop if the height at 'max_next' has reached or exceeded the initial maximum
            if prev_max_height >= values[maxi]:
                break

    return dips  # Return the list of detected dips


def extract_max_dips(entries):
    """
    Extract maximal dips from a list of timestamp tuples (t0, t1).

    Args:
        entries (list of tuples): A sorted list of (t0, t1) tuples, where t0 < t1.

    Returns:
        list: A list of maximal dip entries, each represented as a tuple (t0, t1).
    """
    entries = list(entries)  # call by copy
    # If there are no dips, there cannot be maximal dips
    if not entries:
        return []

    def _filter_max_timestamp(entries, target_t):
        """
        Given a list of timestamp tuples (t0, t1) and a target t0, this function returns the tuple
        with the highest t1 for the given t0, and also returns the filtered list with all entries
        having a t0 less than or equal to the given t0 removed.

        Args:
            entries (list of tuples): A list of (t0, t1) tuples.
            target_t (int): The t0 value for which to find the highest t1.

        Returns:
            tuple: The tuple (t0, t1) with the highest t1 for the given t0.
            list: The filtered list of tuples with all earlier entries removed.
        """
        # Find the tuple with the highest t1 for the given target_t0
        max_t1_entry = max((entry for entry in entries if entry[0] == target_t), key=lambda x: x[1], default=None)

        if max_t1_entry is None:
            # If no matching t0 was found, return None for the tuple and the original list
            return None, entries

        # Filter the list to remove all entries with t0 <= target_t
        filtered_entries = [entry for entry in entries if entry[0] > target_t]

        return max_t1_entry, filtered_entries

    max_dips = []
    target_t = entries[0][0]  # Start with the smallest t0

    while entries:
        max_entry, entries = _filter_max_timestamp(entries, target_t)
        """
        Since I myself forgot how this piece of art works, I recorded the calls and returned values of
        _filter_max_timestamp with a toy example.
        Example:
        >>> extract_max_dips([(0,2),(0,4),(2,4),(4,6),(4,7),(6,8),(7,9)])
        Call _filter_max_timestamp with entries [(0, 2), (0, 4), (2, 4), (4, 6), (4, 7), (6, 8), (7, 9)] and target 0
        returned (0, 4) and [(2, 4), (4, 6), (4, 7), (6, 8), (7, 9)]
        
        Call _filter_max_timestamp with entries [(2, 4), (4, 6), (4, 7), (6, 8), (7, 9)] and target 4
        returned (4, 7) and [(6, 8), (7, 9)]
        
        Call _filter_max_timestamp with entries [(6, 8), (7, 9)] and target 7
        returned (7, 9) and []
        """

        if max_entry is not None:
            max_dips.append(max_entry)
            target_t = max_entry[1]  # Update target_t to the t1 of the found entry
        else:
            # No more entries with the current target_t
            break

    return max_dips


def extract_mdd_from_dip(max_dips, mins, values):
    """
    Extracts Maximum Drawdown (MDD) information for each dip from a list of maximum dips.

    This function calculates the Maximum Drawdown (MDD) for each maximum dip by finding the minimum
    value within the range of each dip and comparing it to the maximum value at the start of the dip.
    It returns a dictionary where each key is a dip, and the value is another dictionary containing
    the MDD value and a vertical line for visualization.

    Parameters:
    - max_dips (list of tuples): A list where each tuple represents a dip. Each tuple contains two elements:
      the start and end indices of the dip (e.g., (start_index, end_index)).
    - mins (list of int): A list of indices where the minima occur.
    - values (list or array): A list or array of values corresponding to the data points.

    Returns:
    - dict: A dictionary where the keys are tuples representing the dips and the values are dictionaries with:
      - "value" (float): The Maximum Drawdown (MDD) as a percentage. Calculated as the difference between
        the maximum value before the dip and the minimum value within the dip, divided by the maximum value
        before the dip.
      - "line" (tuple of tuples): A tuple representing a vertical line for visualization. The first element is
        a tuple (index, value) for the minimum point, and the second element is a tuple (index, value) for the
        maximum value before the dip. This vertical line helps in visualizing the MDD on a plot.

    Example:
    >>> max_dips = [(5, 10), (15, 20)]
    >>> mins = [7, 17]
    >>> values = [10, 20, 30, 25, 30, 28, 20, 18, 22, 25, 30, 35, 40, 38, 36, 30, 25, 22, 20, 18]
    >>> result = _extract_mdd_from_dip(max_dips, mins, values)
    >>> print(result)
    {
        (5, 10): {
            "value": 0.3333,
            "line": ((7, 18), (7, 30))
        },
        (15, 20): {
            "value": 0.25,
            "line": ((17, 22), (17, 30))
        }
    }
    """
    # Convert the indices of minimum points to their corresponding values
    mins_values = np.array(values)[mins]

    # Combine the minimum indices and their values into a list of tuples (index, value)
    min_points = [point for point in zip(mins, mins_values)]

    # Dictionary to store the MDD (Maximum Drawdown) information for each dip
    MDDs = {}

    # Iterate over each maximum dip (represented as a tuple of start and end indices)
    for max_dip in max_dips:
        # Extract all minimum points that lie within the range of the current dip
        mins_in_max_dip = [point for point in min_points if max_dip[0] <= point[0] <= max_dip[1]]

        # Ensure that there is at least one minimum point within the dip range
        assert len(mins_in_max_dip) > 0, f"No minimum found within the dip range {max_dip}"

        # Find the minimum point with the lowest value in the current dip
        min_in_dip = min(mins_in_max_dip, key=lambda x: x[1])

        # Retrieve the maximum value at the start of the dip (before the drawdown occurs)
        max_value_before_dip = values[max_dip[0]]

        # Calculate the Maximum Drawdown (MDD) as a percentage of the max value before the dip
        MDD_value = (max_value_before_dip - min_in_dip[1]) / max_value_before_dip

        # Store the MDD value and the corresponding vertical line in the dictionary
        MDDs[max_dip] = {
            "value": MDD_value,  # MDD value as a percentage
            "line": (min_in_dip, (min_in_dip[0], max_value_before_dip))  # Vertical line for visualization
        }

    # Return the dictionary containing MDD information for each dip
    return MDDs


def smoother(values, threshold=2):
    values = values.copy()  # make it a call by copy function rather than call by reference
    out = [values.pop(0)]  # get first element
    value = values.pop(0)  # get second element
    if abs(value - out[-1]) >= threshold / 100:
        out.append(value)
    else:
        out.append(out[-1])
    for value in values:
        if (abs(value - out[-1]) >= threshold / 100) or (out[-2] < out[-1] < value) or (out[-2] > out[-1] > value):
            out.append(value)
        else:
            out.append(out[-1])

    return out


def get_recovery(y_values, max_dips):
    recovery = {}  # key: position
    # value: dict(recovery= value, line=((),()))
    for b, e in max_dips:
        relative_recovery_difference = y_values[e] / y_values[b]  # degree of resilience # degree is not intuitive!
        # also not intuitive since pos and neg.
        absolute_recovery_difference = y_values[b] - y_values[e]  # resilience error, also not intuitive since pos and

        recovery[e] = dict(relative_recovery=relative_recovery_difference,
                           absolute_recovery=absolute_recovery_difference,
                           line=((e, y_values[b]), (e, y_values[e]))
                           )
    return recovery


def _objective_function(num_segments, x, y, penalty_factor=0.05):
    """
    Objective function for Bayesian Optimization with a penalty on the number of segments.

    This function is used in Bayesian Optimization to find the optimal number of segments
    for piecewise linear fitting. It calculates the total objective by combining the sum of squared
    residuals (SSR) from the Piecewise Linear Fit (PWLF) model and a penalty term based on the number of segments.

    Parameters:
    - num_segments (list or array-like): The number of segments to be used in the Piecewise Linear Fit.
    - x (array-like): The x-values for the piecewise linear fitting.
    - y (array-like): The y-values for the piecewise linear fitting.
    - penalty_factor (float, optional): The factor by which the number of segments is penalized. Default is 0.05.

    Returns:
    - float: The total objective value, which is the sum of SSR and the penalty term.
    """
    num_segments = int(num_segments[0])  # Convert list to integer

    # Fit the Piecewise Linear Fit model
    pwlf_model = pwlf.PiecewiseLinFit(x, y)
    pwlf_model.fit(num_segments)

    # Calculate the sum of squared residuals
    ssr = pwlf_model.ssr

    # Penalty term: You can adjust the factor as needed
    penalty = penalty_factor * num_segments

    # Total objective is SSR plus penalty
    total_objective = ssr + penalty

    return total_objective


def _perform_bayesian_optimization(x_values, y_values, dimensions=10, penalty_factor=0.05):
    """
    Perform Bayesian Optimization to find the optimal number of segments
    for piecewise linear fitting.

    Parameters:
    - x_values (array-like): The x-values for the data series.
    - y_values (array-like): The y-values for the data series.
    - dimensions (integer): Maximal number of segments
    - penalty_factor (float, optional): The factor by which the number of segments is penalized. Default is 0.05.

    Returns:
    - int: The optimal number of segments found by the optimization.

    Example:
    >>> x = [0, 1, 2, 3, 4]
    >>> y = [10, 15, 20, 25, 30]
    >>> dimensions = 20  # Specify the range for the number of segments
    >>> _perform_bayesian_optimization(x, y, dimensions=dimensions)
    5
    """
    result = gp_minimize(
        func=lambda num_segments: _objective_function(num_segments, x_values, y_values, penalty_factor=penalty_factor),
        dimensions=[Integer(1, dimensions)],  # Adjust bounds as needed
        n_calls=int(1.5 * dimensions),  # 1.5 times the number of dimensions
        random_state=42,
        n_jobs=-1
    )
    return int(result.x[0])
