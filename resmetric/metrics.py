import numpy as np
import math
from scipy.signal import find_peaks

import pwlf
from skopt import gp_minimize
from skopt.space import Integer


def calculate_kernel_auc(y_values, kernel='uniform', lambda_weight=None, half_life=None, custom_kernel=None):
    """
    Calculate the Area Under the Curve (AUC) using a kernel-weighted approach.
    Supports uniform, inverse, and exponential decay kernels, or a custom kernel function.

    Parameters:
    - y_values (list): Input list of y-values representing the data series.
    - kernel (str): The kernel to use for weighting ('uniform', 'inverse', 'exp', 'custom').
    - lambda_weight (float): The lambda parameter for the exponential decay kernel.
    - half_life (float): The half-life for the exponential decay kernel. If provided, it overrides `lambda_weight`.
    - custom_kernel (function): A custom kernel function (anonymous lambda) that takes a single argument `d`.

    Returns:
    - auc_values (list): List of AUC values calculated at each point in the series.

    Raises:
    - ValueError: If an unsupported kernel is selected or if required parameters are missing.
    """

    def _half_life_to_lambda(half_life):
        """Convert half-life to decay parameter lambda."""
        return math.log(2) / half_life

    # Define kernel functions for different kernel types
    def uniform_kernel(d):
        # Uniform kernel: Equal weight for all data points
        return 1

    def inverse_kernel(d):
        # Inverse kernel: Weight decreases as the distance increases
        return 1 / (d + 1)

    def exp_kernel(d, lambda_weight):
        # Exponential decay kernel: Weight decreases exponentially with distance
        return np.exp(-lambda_weight * d)

    if half_life is not None:
        lambda_weight = _half_life_to_lambda(half_life)

    # Select the appropriate kernel function based on user input
    if kernel == 'uniform':
        kernel_func = uniform_kernel
    elif kernel == 'inverse':
        kernel_func = inverse_kernel
    elif kernel == 'exp':
        if lambda_weight is None:
            # Raise error if exponential kernel is selected but no lambda is provided
            raise ValueError("lambda_weight or half_life must be provided for exponential kernel.")
        kernel_func = lambda d: exp_kernel(d, lambda_weight)
    elif kernel == 'custom':
        if custom_kernel is None:
            # Raise error if custom kernel is selected but no function is provided
            raise ValueError("A custom kernel function must be provided when using the 'custom' kernel option.")
        kernel_func = custom_kernel
    else:
        # Raise error if an unsupported kernel is selected
        raise ValueError(
            f"Unsupported kernel '{kernel}'. Supported kernels are 'uniform', 'inverse', 'exp', or 'custom'.")

    auc_values = []  # List to store the calculated AUC values for time series

    # Loop through each point in the data series
    for i in range(1, len(y_values)):
        weighted_sum = 0  # Initialize weighted sum for the current AUC calculation
        kernel_sum = 0  # Initialize sum of kernel weights

        # Calculate the AUC using trapezoidal rule with kernel weighting
        for j in range(1, i + 1):
            weight = kernel_func(i - j)  # Calculate the weight using the selected kernel
            trapezoid_area = ((y_values[j - 1] + y_values[j]) / 2) * weight  # Calculate trapezoid area
            weighted_sum += trapezoid_area  # Add to weighted sum
            kernel_sum += weight  # Add to kernel weight sum

        # Store the normalized AUC value in the list
        auc_values.append(weighted_sum / kernel_sum)

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
    # Add padding with -1 at the beginning and end of y_values
    padded_y_values = np.concatenate(([-2], y_values, [-2]))

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


def extract_max_dips_based_on_threshold(values, threshold):
    """
    Identifies the dips in the `values` array where the values fall below the `threshold`
    and then rise above it again. A dip is recorded only if it starts above the threshold,
    falls below, and then returns above the threshold.

    Parameters:
    values (list or np.ndarray): The array of values to analyze.
    threshold (float): The threshold below which a dip is identified.

    Returns:
    list of tuples: A list of (start_index, end_index) tuples, each representing the
                    start and end indices of a dip.

    Example:
    >>> values = [9, 12, 15, 8, 5, 11, 14, 7, 20, 4]
    >>> threshold = 10
    >>> extract_max_dips_based_on_threshold(values, threshold)
    [(3, 5), (7, 8)]
    """

    # Convert values to a NumPy array and create a binary array where values are above the threshold
    values = np.array(values)
    below_list = ((values - (threshold / 100)) > 0).astype(int)

    def _sign_changes_series(series):
        """
        Calculates sign changes in the binary series.

        Parameters:
        series (np.ndarray): The binary series indicating whether values are above (1) or below (0) the threshold.

        Returns:
        np.ndarray: An array of sign changes where 1 indicates an increase and -1 indicates a decrease.
        """
        sign_changes_list = [0]  # No sign change at index 0 possible
        for i in range(1, len(series)):
            sign_changes_list.append(series[i] - series[i - 1])
        return np.array(sign_changes_list)

    # Detect sign changes in the binary series
    sign_changes = _sign_changes_series(below_list)

    # Find indices where the series changes from above to below the threshold
    minus_one = np.argwhere(sign_changes == -1).flatten()

    # Find indices where the series changes from below to above the threshold
    plus_one = np.argwhere(sign_changes == 1).flatten()

    # Align start (minus_one) and end (plus_one) indices
    if len(minus_one) > 0 and len(plus_one) > 0 and minus_one[0] > plus_one[0]:
        plus_one = plus_one[1:]

    # Adjust minus_one indices to correct positions
    minus_one -= 1

    # Pair start and end indices of dips
    dips = list(zip(minus_one, plus_one))

    return dips


def extract_max_dips_based_on_maxs(entries):
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
        >>> extract_max_dips_based_on_maxs([(0,2),(0,4),(2,4),(4,6),(4,7),(6,8),(7,9)])
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


def extract_mdd_from_dip(max_dips, values):
    """
    Extracts Maximum Drawdown (MDD) information for each dip from a list of maximum dips.

    This function calculates the Maximum Drawdown (MDD) for each maximum dip by finding the minimum
    value within the range of each dip and comparing it to the maximum value at the start of the dip.
    It returns a dictionary where each key is a dip, and the value is another dictionary containing
    the MDD value and a vertical line for visualization.

    Parameters:
    - max_dips (list of tuples): A list where each tuple represents a dip. Each tuple contains two elements:
      the start and end indices of the dip (e.g., (start_index, end_index)).
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
    >>> values = [10, 20, 30, 25, 30, 28, 20, 18, 22, 25, 30, 35, 40, 38, 36, 30, 25, 22, 20, 18]
    >>> result = extract_mdd_from_dip(max_dips, mins, values)
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

    # Dictionary to store the MDD (Maximum Drawdown) information for each dip
    MDDs = {}

    # Iterate over each maximum dip (represented as a tuple of start and end indices)
    for start, end in max_dips:
        # Extract the y-values within the dip range
        dip_values = np.array(values[start:end + 1])

        # Detect local minima within the dip by inverting the values
        minima_indices = detect_peaks(-dip_values)

        # Ensure that there is at least one minimum point within the dip range
        assert len(minima_indices) > 0, f"No minimum found within the dip range {(start, end)}"

        # Identify the index and value of the local minimum within the dip
        min_index_in_dip = minima_indices[np.argmin(dip_values[minima_indices])]
        min_value = dip_values[min_index_in_dip]
        min_index = start + min_index_in_dip

        # Step 3: Find the local maximum in the range before the local minimum
        # Detect local maxima
        maxima_indices = detect_peaks(dip_values[:min_index_in_dip])
        # Ensure that there is at least one minimum point within the dip range
        if not len(maxima_indices) > 0:
            # This is the edge case when a dip consists of a sudden positive change
            # The minimum is the first point and the values of the dip are monotonically increasing.
            # f"No maximum found within the dip range {(start, end)} before {min_index}"
            MDDs[(start, end)] = {
                "value": 0,  # MDD value as a percentage
                "line": ((min_index, min_value), (min_index, min_value))  # Vertical line for visualization
            }
        else:
            max_before_min_in_dip = np.argmax(dip_values[maxima_indices])
            max_value = dip_values[max_before_min_in_dip]
            # Calculate the Maximum Drawdown (MDD) as a percentage of the max value before the dip
            MDD_value = (max_value - min_value) / max_value

            # Store the MDD value and the corresponding vertical line in the dictionary
            MDDs[(start, end)] = {
                "value": MDD_value,  # MDD value as a percentage
                "line": ((min_index, min_value), (min_index, max_value))  # Vertical line for visualization
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


def get_recovery(y_values, max_dips, algorithm='adaptive_capacity'):
    """
    Calculate recovery metrics based on the maximum dips using the specified algorithm.

    Parameters:
    - y_values (list or array): The y-values for the data points.
    - max_dips (list of tuples): List of tuples representing the dips with (start_index, end_index).
    - algorithm (str): The recovery level algorithm to use. Can be:
      - 'adaptive_capacity' (default): Ratio of new steady state to prior steady state value (Q(t_ns) / Q(t_0)).
      - 'recovery_ability': Relative recovery as
        ((Q(t_ns) - Q(t_r)) / (Q(t_0) - Q(t_r))), where Q(t_r)
        is the local minimum within the dip.

    Returns:
    - dict: Dictionary where keys are end indices of the dips and values are dictionaries containing:
      - 'relative_recovery': The relative recovery metric. (recovery level)
      - 'absolute_recovery': The absolute recovery metric. (recovery level)
      - 'line': A tuple representing a vertical line for visualization (e.g., (x_value, y_min, y_max)).

    Raises:
    - ValueError: If `algorithm` is not one of the valid options ('adaptive_capacity' or 'recovery_ability').
    - ValueError: If any dip tuple has invalid indices (b >= e, b < 0, or e >= len(y_values)).
    """
    # Validate the algorithm parameter
    valid_algorithms = ['adaptive_capacity', 'recovery_ability']
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Choose one of {valid_algorithms}.")

    recovery = {}  # key: position
    # value: dict(recovery= value, line=((),()))
    for b, e in max_dips:
        if b >= e or b < 0 or e >= len(y_values):
            raise ValueError(f"Invalid dip indices: ({b}, {e})")

        if algorithm == 'adaptive_capacity':
            # Adaptive Capacity: Q(t_ns) / Q(t_0)
            relative_recovery_difference = y_values[e] / y_values[b]  # degree of resilience # degree is not intuitive!
            # also not intuitive since pos and neg.
        elif algorithm == 'recovery_ability':
            # Recovery Ability: (Q(t_ns) - Q(t_r)) / (Q(t_0) - Q(t_r))
            local_min = min(y_values[b:e + 1])
            relative_recovery_difference = abs((y_values[e] - local_min) / (y_values[b] - local_min))

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


def resilience_over_time(dips_resilience):
    """
    Calculates the differential quotient of resilience metrics over time from a dictionary of dips and their corresponding resilience metrics.

    Parameters:
    - dips_resilience (dict): A dictionary where keys are tuples representing dips (start, end) and values are dictionaries with resilience metrics.
      Example:
      {
        (3, 7): {"robustness": 0.5, "recovery": 0.8},
        (10, 42): {"robustness": 0.7, "recovery": 0.5},
        (69, 75): {"robustness": 0.8, "recovery": 0.7}
      }

    Returns:
    - dict: A dictionary with metrics as keys and their differential quotients and overall mean values.
      Example:
      {
        "robustness": {
            "diff_q": [(42, 0.2), (75, 0.1)],
            "overall": 0.15
        },
        "recovery": {
            "diff_q": [(42, -0.3), (75, 0.2)],
            "overall": -0.05
        }
      }
    """
    # Initialize a dictionary to store results
    results = {}

    # Extract metrics from the first entry
    if not dips_resilience:
        return results  # Return empty result if no data

    # Get the metrics from the first entry
    first_dip = next(iter(dips_resilience))
    available_metrics = set(dips_resilience[first_dip].keys())

    # Check that these metrics are present in all entries
    for metrics_dict in dips_resilience.values():
        available_metrics.intersection_update(metrics_dict.keys())

    # Calculate antifragility under u for each metric
    for metric in available_metrics:
        u = [v[metric] for v in dips_resilience.values()]
        if not len(u) >= 2:
            results[metric] = None
            continue

        # Calculate M_u
        # Fraction of antifragile sequence fulfillment
        M_u = sum(1 for k in range(1, len(u)) if u[k] - u[k - 1] >= 0) / (len(u) - 1)

        # If constant u
        if all(x == u[0] for x in u):
            I_u = 0
            A_u = 1

        else:
            I_k = []
            # Calculate I_k values
            for k in range(1, len(u)):
                if u[k - 1] != 0:  # Avoid division by zero
                    I_k.append((u[k] - u[k - 1]) / u[k - 1])
                else:
                    I_k.append(0)  # Substitute with 0 for undefined I_k

            # Calculate I_u (average rate of improvement)
            I_u = np.mean(I_k)

            # Calculate A_u
            # Antifragility under u
            A_u = sum(max(0, i) for i in I_k) / (sum(abs(i) for i in I_k))

        # Calculate the degree of antifragility under u: alpha_u
        alpha_u = A_u if A_u < 1 else 1 + I_u

        # Store results
        results[metric] = {
            "M_u": M_u,
            "I_u": I_u,
            "A_u": A_u,
            "alpha_u": alpha_u
        }

    return results


def get_dip_auc(y_values, max_dips):
    """
    Calculate the Area Under Curve (AUC) for each dip defined in `max_dips`.

    This function computes the AUC for segments of the `y_values` array corresponding to each dip range specified in `max_dips`.
    It uses the `calculate_kernel_auc` function to obtain the AUC value for each segment.

    Parameters:
    - y_values (list or array-like): A list or array of numerical values representing the data points.
    - max_dips (list of tuples): A list of tuples, where each tuple contains two elements (start_index, end_index)
      defining the range of the dip. Each range specifies the segment of `y_values` to calculate the AUC for.

    Returns:
    - dict: A dictionary where each key is a tuple representing a dip (start, end), and the value is the AUC
      of that segment. The AUC is calculated using the `calculate_kernel_auc` function applied to the segment.

    Example:
    >>> y_values = [10, 20, 30, 25, 30, 28, 20, 18, 22, 25, 30, 35, 40, 38, 36, 30, 25, 22, 20, 18]
    >>> max_dips = [(5, 10), (15, 20)]
    >>> result = get_dip_auc(y_values, max_dips)
    >>> print(result)
    {
        (5, 10): 10.5,  # Example AUC values (not accurate)
        (15, 20): 9.2
    }
    """
    # Dictionary to store the AUC for each dip
    dip_auc_info = {}

    # Iterate over each dip defined by its start and end indices
    for dip in max_dips:
        # Extract the segment of y_values corresponding to the current dip
        segment = y_values[dip[0]:dip[1] + 1]
        # Calculate the AUC for the segment and store it in the dictionary
        dip_auc_info[dip] = calculate_kernel_auc(segment)[-1]

    return dip_auc_info


def get_max_dip_integrated_resilience_metric(y_values, max_dips):
    """
    Calculate the Integrated Resilience Metric for each dip defined in `max_dips`.
    (cf. Sansavini, https://doi.org/10.1007/978-94-024-1123-2_6, Chapter 6, formula 6.12,
    formula fixed to ((TAPL +1) ** -1)) cf. artefact publication)

    This function computes the IRM for segments of the `y_values` array corresponding
    to each dip range specified in `max_dips`.

    Parameters
    ----------
    y_values : list or array-like
        A list or array of numerical values representing the data points.
    max_dips : list of tuples
        A list of tuples, where each tuple contains two elements (t_d, t_ns)
        defining the range of the dip. Each range specifies the segment of `y_values` to calculate the IRM for.

    Returns
    -------
    dict
        A dictionary where each key is a tuple representing a dip (t_d, t_ns), and the value is the IRM
        of that segment.

    Example
    -------
    >>> y_values = [10, 20, 30, 25, 30, 28, 20, 18, 22, 25, 30, 35, 40, 38, 36, 30, 25, 22, 20, 18]
    >>> max_dips = [(5, 10), (15, 20)]
    >>> result = get_max_dip_integrated_resilience_metric(y_values, max_dips)
    >>> print(result)
    {
        (5, 10): 8.3,  # Example GR values (not accurate)
        (15, 20): 7.1
    }
    """
    irm_info = {}

    for dip in max_dips:
        t_d, t_ns = dip
        segment = y_values[t_d:t_ns + 1]

        # Extract the relevant values
        Q_t_d = segment[0]  # Value at the start of the dip (Q_{t_d})
        Q_t_r = min(segment)  # Robustness: Minimum value within the dip (Q_{t_r})
        Q_t_ns = segment[-1]  # Value at the end of the dip (Q_{t_ns})

        # Find the index of the minimum value within the dip segment
        min_index = np.argmin(segment) if isinstance(segment, np.ndarray) else segment.index(Q_t_r)
        t_r = t_d + min_index  # The index where the minimum occurs

        # Assert that t_r is not equal to t_d to avoid division by zero in RAPI_DP calculation
        assert t_r != t_d, "t_r must not be equal to t_d. Check your dip detection algorithm."

        # Assert that Q_t_d - Q_t_r is not zero to avoid division by zero in RA calculation
        assert (Q_t_d - Q_t_r) != 0, "Q_t_d - Q_t_r must not be zero. Check your dip detection algorithm."

        # Calculate RAPI
        RAPI_DP = (Q_t_d - Q_t_r) / (t_r - t_d)  # Rate of performance decline
        RAPI_RP = (Q_t_ns - Q_t_r) / (t_ns - t_r) if t_ns != t_r else 0  # Rate of performance recovery

        # Time-Averaged Performance Loss (TAPL)
        TAPL = calculate_kernel_auc([Q_t_d - value for value in segment])[-1]

        # Recovery Ability (RA) (One form of recovery level)
        RA = get_recovery(segment, [(0, len(segment) - 1)],
                          algorithm='recovery_ability')[len(segment) - 1]['relative_recovery']

        # Calculate IRM
        # Variant by Sansavini (Does not consider cases where TAPL < 0 (fatal)
        # In the original, GR is the symbol for IRM
        # IRM = Q_t_r * (RAPI_RP / RAPI_DP) * (TAPL ** -1) * RA

        # Fixed version
        IRM = Q_t_r * (RAPI_RP / RAPI_DP) * ((TAPL + 1) ** -1) * RA

        irm_info[dip] = IRM

    return irm_info


def mdd_to_robustness(mdd: float) -> float:
    """
    Convert Maximum Drawdown (MDD) to Robustness metric.

    The robustness metric provides a measure of system stability by quantifying
    how resilient a system is to losses represented by the Maximum Drawdown.

    Parameters:
        mdd (float):
            The Maximum Drawdown value, representing the largest percentage drop
            from a peak to a trough. Expected to be a value between 0 and 1,
            where 0 indicates no drawdown and 1 indicates a 100% loss.

    Returns:
        float:
            The robustness metric calculated as (1 - mdd). The result ranges between
            0 and 1, where a value closer to 1 indicates higher robustness.

    Raises:
        ValueError:
            If the input `mdd` is not between 0 and 1.

    Examples:
        >>> mdd_to_robustness(0.2)
        0.8

        >>> mdd_to_robustness(0.0)
        1.0

        >>> mdd_to_robustness(1.0)
        0.0

    """
    if not 0 <= mdd <= 1:
        raise ValueError("mdd must be between 0 and 1 inclusive.")
    return 1 - mdd


def dip_to_recovery_rate(dip: tuple) -> float:
    """
    Calculate the inverse of the recovery time from a dip event.

    The recovery rate is defined as the inverse of the time it takes to recover
    from a dip, which is calculated as the difference between the end time (t1)
    and the start time (t0). A higher recovery rate indicates a quicker recovery.

    Parameters:
        dip (tuple):
            A tuple containing two time points (t0, t1), where t0 is the time
            when the dip starts and t1 is the time when recovery occurs. Both
            t0 and t1 should be numeric values, and t1 must be greater than t0.

    Returns:
        float:
            The recovery rate, calculated as 1 / (t1 - t0). The result is a
            positive number, where a larger value indicates a faster recovery.

    Raises:
        ValueError:
            If t1 is not greater than t0, or if the tuple does not contain exactly
            two elements.

    Examples:
        >>> dip_to_recovery_rate((2, 5))
        0.3333333333333333

        >>> dip_to_recovery_rate((0, 10))
        0.1

        >>> dip_to_recovery_rate((5, 5))
        ValueError: t1 must be greater than t0.

    """
    # Ensure the tuple contains exactly two elements
    if len(dip) != 2:
        raise ValueError("dip must be a tuple with exactly two elements (t0, t1).")

    t0, t1 = dip

    # Ensure that t1 is greater than t0
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0.")

    # Calculate the recovery rate
    return 1 / (t1 - t0)
