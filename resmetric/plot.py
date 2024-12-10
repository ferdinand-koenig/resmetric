from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pwlf
import copy
from .metrics import (
    detect_peaks,
    _get_dips,
    extract_max_dips_based_on_maxs,
    extract_mdd_from_dip,
    get_recovery,
    calculate_kernel_auc,
    time_below_threshold,
    count_dibs_below_threshold_series,
    calculate_max_drawdown,
    _find_next_smaller,
    smoother,
    _perform_bayesian_optimization,
    _make_color_pale_hex,
    resilience_over_time,
    get_dip_auc,
    mdd_to_robustness,
    dip_to_recovery_rate,
    get_max_dip_integrated_resilience_metric,
    extract_max_dips_based_on_threshold
)

# Global variable to control printing
verbose = True

# Text font of a bar
BAR_TEXT_FONT = dict(
    size=40,  # Set the font size
    color='black',  # Set the font color to black
)


def set_verbose(enabled):
    """Enable or disable verbose output."""
    global verbose
    verbose = enabled


def vprint(*args, **kwargs):
    """Print only if verbose is enabled."""
    if verbose:
        print(*args, **kwargs)


def create_plot_from_data(json_str, **kwargs):
    """
    Generate a Plotly figure from a JSON-encoded Plotly figure with optional
    traces and analyses based on existing traces.

    Parameters
    ----------
    json_str : str
        JSON string containing the figure data.
    **kwargs : dict, optional
        Optional keyword arguments to include or exclude specific traces and
        analyses:

        - include_smooth_criminals (bool): Preprocess the series and smooth with a threshold-based update filter.
          (Hee-Hee! Ow!)

        - include_auc (bool): Include AUC-related traces. (AUC divided by the length of the time frame and
          different kernels applied)
        - include_count_below_thresh (bool): Include traces counting dips below
          the threshold.
        - include_time_below_thresh (bool): Include traces accumulating time
          below the threshold.
        - threshold (float): Threshold in percent for count and time traces (default is 80[%]).
        - include_dips (bool): Include detected dips.
        - include_draw_downs_shapes (bool): Include shapes of local draw-downs.
        - include_draw_downs_traces (bool): Include traces representing the
          relative loss at each point in the time series, calculated as the
          difference between the current value and the highest value reached
          up to that point, divided by that highest value.
        - include_derivatives (bool): Include derivatives traces.

        - dip_detection_algorithm (str): Specifies the dip detection algorithm to use.
          It can be 'max_dips' (default), 'threshold_dips', 'manual_dips', 'lin_reg_dips' (the last requires
          include_lin_reg).
        - manual_dips (list of tuples or None): If 'manual_dips' is selected as the dip detection algorithm,
          this should be a list of tuples specifying the manual dips.
        - include_lin_reg (bool or float): Include linear regression traces. Optionally float for threshold of slope.
          Slopes above the absolute value are discarded. Defaults to 0.5% (for value set to True). (Unitless! not in %)
          It is possible to pass math.inf. See also `no_lin_reg_prepro`
        - no_lin_reg_prepro (bool): include_lin_reg automatically preprocesses and updates the series. If you do
          not wish this, set this flag to True

        - include_dip_auc (bool): Include AUC bars for the AUC of one maximal dip
          (AUC devided by the length of the time frame)
        - include_bars (bool): Include bars for robustness, recovery level and recovery time.
        - include_irm (bool): Include the Integrated Resilience Metric
          (cf. Sansavini, https://doi.org/10.1007/978-94-024-1123-2_6, Chapter 6, formula 6.12
          formula fixed to ((TAPL +1) ** -1)) cf. artefact publication)
          Requires kwarg recovery_algorithm='recovery_ability'.
        - recovery_algorithm (str or None): Decides the recovery level algorithm.
          Can either be 'adaptive_capacity' (default) or 'recovery_ability'.
          The first one is the ratio of new to prior steady state's value (Q(t_ns) / Q(t_0)).
          The last one is abs((Q(t_ns) - Q(t_r))/ (Q(t_0) - Q(t_r)))
          where Q(t_r) is the local minimum within a dip (Robustness).

        - calc_res_over_time (bool): Calculate the differential quotient for every per-dip Resilience-Related Trace.

        - penalty_factor (float): Penalty factor for Bayesian Optimization
          (default is 0.05).
        - dimensions (int): Dimensions for Bayesian Optimization (default is 10)
          (Max. N. of segments for lin. reg.).
        - weighted_auc_half_life (float): Half-life for weighted AUC calculation
          (default is 2).
        - smoother_threshold (int): Threshold for smoother function in percent
          (default is 2[%]).

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly Figure object with the specified traces and analyses included.
    """
    # Set default dip detection algorithm to 'max_dips'
    dip_detection_algorithm = kwargs.get('dip_detection_algorithm', None)
    recovery_algorithm = kwargs.get('recovery_algorithm', 'adaptive_capacity')

    # # Validate the include_irm parameter
    # if kwargs.get('include_irm') and recovery_algorithm != 'recovery_ability':
    #     raise ValueError(
    #         "The 'include_irm' option requires the 'recovery_algorithm' to be set to 'recovery_ability'. "
    #         "Please set 'recovery_algorithm' to 'recovery_ability' to include the Integrated Resilience Metric."
    #     )

    # Convert JSON string to Plotly figure
    fig = pio.from_json(json_str)

    # Extract data series and initialize trace lists
    series = fig.data
    smooth_criminals = []
    auc_traces = []
    time_below_thresh_traces = []
    count_below_thresh_traces = []
    threshold_line = []
    drawdown_traces = []
    draw_down_shapes = []
    maximal_dips_shapes = []
    maximal_dips_bars = []
    dips_horizontal_shapes = []
    derivative_traces = []
    lin_reg_traces = []
    antifrag_diff_qu_traces = []
    dip_auc_bars = []
    irm_bars = []

    # Retrieve optional arguments with defaults
    threshold = kwargs.get('threshold', 80)
    penalty_factor = kwargs.get('penalty_factor', 0.05)
    dimensions = kwargs.get('dimensions', 10)
    weighted_auc_half_life = kwargs.get('weighted_auc_half_life', 2)
    smoother_threshold = kwargs.get('smoother_threshold', 2)

    if kwargs.get('include_lin_reg', False) is not False:
        lin_reg_threshold = .5e-2 if kwargs['include_lin_reg'] is True else abs(kwargs['include_lin_reg'])
        vprint(f'Using threshold {lin_reg_threshold}')

    # Initialize variables for global x limits
    global_x_min = float('inf')
    global_x_max = float('-inf')

    for i, s in enumerate(series):
        s.update(
            mode='lines+markers',
            marker=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i])),
            line=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i]))
        )

        y_values = s.y
        x_values = s.x if s.x is not None else np.arange(len(y_values))  # Assuming x-values are indices

        # Update global x limits
        global_x_min = min(global_x_min, x_values[0])
        global_x_max = max(global_x_max, x_values[-1])

        ################################################
        # Preprocessing
        # Append smooth criminal traces if requested
        if kwargs.get('include_smooth_criminals'):
            y_values = smoother(list(y_values), threshold=smoother_threshold)
            smooth_criminals.append(go.Scatter(
                name=f"Smoothed {s.name}",
                y=y_values,
                mode='lines+markers',
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                legendgroup=f'Smoothed {s.name}'
            ))

        # [T-Dip] Fit the piecewise linear model and add to traces if requested
        if kwargs.get('include_lin_reg', False) is not False:
            # Suppress only UserWarnings temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                # Perform Bayesian Optimization to find the optimal number of segments
                vprint(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating linear regression"
                       f" of series {i + 1} of {len(series)}")
                optimal_segments = _perform_bayesian_optimization(x_values, y_values,
                                                                  penalty_factor=penalty_factor,
                                                                  dimensions=dimensions)
                pwlf_model = pwlf.PiecewiseLinFit(x_values, y_values)
                pwlf_model.fit(optimal_segments)

                # Using the breakpoints provided by the package turned out to be a fool's errand.
                # The problem is:
                #   - a linear regression sometimes until t and the next one starts at t + 1
                #     (generating a gap of [t, t+1])
                #   - this, however is not a rule so these gaps might not be generated
                #   - also, a segment might be of size 2 (exactly two points)
                # With the provided breakpoints, it is not trivial to derive the start and endpoint of a segment
                # Solution: Use the model's function to map the points onto the segments (predict)
                # Then calculate the slopes directly from there (diff. quotient)
                # The indices where the slope changes is your breakpoint
                # Notice: Due to this behavior, the actual number of segments might be larger than `optimal_segments`

                # Extract breakpoints and slopes
                # slopes = pwlf_model.calc_slopes()
                # breakpoints = pwlf_model.fit_breaks

                # for j in range(len(breakpoints)):
                #     try:
                #         next_bigger = round(breakpoints[j + 1])
                #     except:
                #         next_bigger = -1
                #     this = round(breakpoints[j])
                #     breakpoints[j] = this if not next_bigger == this else this - 1

                # Extract start and end points of each segment
                # segments = []
                # for j in range(len(breakpoints) - 1):
                #     start_x = breakpoints[j] + 1 if breakpoints[j] != 0 else 0
                #     end_x = breakpoints[j + 1]
                #     start_y = pwlf_model.predict(start_x)[0]
                #     end_y = pwlf_model.predict(end_x)[0]
                #     # start_y = y_hat[start_x]
                #     # end_y = y_hat[end_x]
                #     slope = slopes[j]
                #     segments.append({
                #         'start_point': (start_x, start_y),
                #         'end_point': (end_x, end_y),
                #         'slope': slope
                #     })

                pwlf_y = pwlf_model.predict(x_values)
                # differential quotient
                diff_q = [float(pwlf_y[i + 1]) - float(pwlf_y[i]) for i in range(len(pwlf_y) - 1)]
                diff_q = [round(num, 6) for num in diff_q]  # numerical float issues ==> solved by rounding to 6 digits

                # Indices of true breakpoints
                indices = [i for i in range(1, len(diff_q)) if diff_q[i] != diff_q[i - 1]]
                indices.insert(0, 0)
                slopes = [diff_q[index] for index in indices]
                indices.append(len(x_values) - 1)

                segments = [{
                    'start_point': (indices[i], pwlf_y[indices[i]]),
                    'end_point': (indices[i + 1], pwlf_y[indices[i + 1]]),
                    'slope': slope
                } for i, slope in enumerate(slopes)]

                filtered_segments = [seg for seg in segments if abs(seg['slope']) < lin_reg_threshold]

                for segment in filtered_segments:
                    start_point = segment['start_point']
                    end_point = segment['end_point']
                    lin_reg_traces.append(
                        go.Scatter(
                            x=[start_point[0], end_point[0]],
                            y=[start_point[1], end_point[1]],
                            mode='lines',
                            line=dict(color=fig.layout.template.layout.colorway[i]),
                            name=f'PWLF ({optimal_segments} Segments) - {s.name}',
                            # name=f'PWLF (x Segments) - {s.name}',
                            legendgroup=f'PWLF ({optimal_segments} ({len(slopes)}) Segments) - {s.name}'
                        )
                    )

            if not kwargs.get('no_lin_reg_prepro'):
                # copy old series
                lin_reg_traces.append(copy.deepcopy(s))
                s.update(
                    name=s.name + ' (prepro - lin reg)'
                )
                # update (smooth) the current
                y_values = pwlf_y
                s.y = y_values

        ################################################
        # [T-Ag] Handle all the agnostic features
        # [T-Ag] Append AUC-related traces if requested
        if kwargs.get('include_auc'):
            auc_values = calculate_kernel_auc(y_values, kernel='uniform')
            auc_traces.append(go.Scatter(
                name=f"AUC {s.name}",
                legendgroup=f"AUC {s.name}",
                x=np.arange(1, len(auc_values) + 1),
                y=auc_values,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

            auc_values_exp = calculate_kernel_auc(y_values, kernel='exp', half_life=weighted_auc_half_life)
            auc_traces.append(go.Scatter(
                name=f"AUC-exp {s.name}",
                legendgroup=f"AUC-exp {s.name}",
                x=np.arange(1, len(auc_values_exp) + 1),
                y=auc_values_exp,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

            auc_values_inv = calculate_kernel_auc(y_values, kernel='inverse')
            auc_traces.append(go.Scatter(
                name=f"AUC-inv {s.name}",
                legendgroup=f"AUC-inv {s.name}",
                x=np.arange(1, len(auc_values_inv) + 1),
                y=auc_values_inv,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

        # [T-Ag] Append count and time traces if requested
        if kwargs.get('include_count_below_thresh'):
            count_below_thresh_traces.append(go.Scatter(
                y=count_dibs_below_threshold_series(y_values, threshold),
                name=f"Count below {threshold}% - {s.name}",
                legendgroup=f"Count below {threshold}% - {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y3'
            ))

        if kwargs.get('include_time_below_thresh'):
            time_below_thresh_traces.append(go.Scatter(
                y=time_below_threshold(y_values, threshold),
                name=f"Time below {threshold}% - {s.name}",
                legendgroup=f"Time below {threshold}% - {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y2'
            ))

        # [T-Ag] Append derivative traces if requested (Experimental)
        if kwargs.get('include_derivatives'):
            # Compute 1st and 2nd derivatives
            first_derivative = np.gradient(y_values)
            second_derivative = np.gradient(first_derivative)

            derivative_traces.extend([
                go.Scatter(
                    y=first_derivative,
                    mode='lines+markers',
                    marker=dict(symbol='diamond-dot', color=fig.layout.template.layout.colorway[i]),
                    name=f'1st derivative - {s.name}',
                    legendgroup=f'1st derivative - {s.name}',
                ),
                go.Scatter(
                    y=second_derivative,
                    mode='lines+markers',
                    marker=dict(symbol='diamond-dot', color=fig.layout.template.layout.colorway[i]),
                    line=dict(dash='dot'),
                    name=f'2nd derivative - {s.name}',
                    legendgroup=f'2nd derivative - {s.name}',
                )
            ])

        ###############
        # Calculate Dips (Local dips). This is not one shock / resilience case!
        # Required for include_dips and max_dips dip detection algorithm
        maxs = detect_peaks(np.array(y_values))
        dips = _get_dips(y_values, maxs=maxs)

        ###############
        # [Advanced][T-Ag] Handle all the advanced agnostic features
        # Append traces for dips and drawdowns if requested
        if kwargs.get('include_dips'):
            for b, e in dips:
                dips_horizontal_shapes.append(
                    go.Scatter(
                        x=[b, e],
                        y=[y_values[b], y_values[b]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                        name=f'Dip Line {s.name}',
                        legendgroup=f'Dip Line {s.name}'
                    )
                )
            dips_horizontal_shapes.append(
                go.Scatter(
                    x=[e for _, e in dips],
                    y=[y_values[s] for s, _ in dips],
                    mode='markers',
                    marker=dict(symbol='x', color=fig.layout.template.layout.colorway[i]),
                    name=f'Dips {s.name}',
                    legendgroup=f'Dip Line {s.name}',
                ))

        if kwargs.get('include_draw_downs_shapes'):
            mins = detect_peaks(-np.array(y_values))
            for mini in mins:
                next_smaller = _find_next_smaller(maxs, mini)
                if next_smaller is None:
                    continue
                draw_down_shapes.append(
                    go.Scatter(
                        x=[mini, mini],
                        y=[y_values[mini], y_values[next_smaller]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i], width=2),
                        name=f'Draw Downs - {s.name}',
                        legendgroup=f'Draw Downs - {s.name}',
                    )
                )

        # Append drawdown traces if requested
        if kwargs.get('include_draw_downs_traces'):
            drawdown_traces.append(go.Scatter(
                y=calculate_max_drawdown(y_values)[1],
                name=f"Drawdown Trace - {s.name}",
                legendgroup=f"Drawdown Trace- {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y3'
            ))

            ################################################

        if (kwargs.get('include_bars') or kwargs.get('include_dip_auc') or kwargs.get('include_irm')
                or dip_detection_algorithm):
            if not dip_detection_algorithm:
                dip_detection_algorithm = 'max_dips'
            # [T-Dip] Dip Detection
            # MaxDips Detection (Detects with the help of peaks)
            if dip_detection_algorithm == 'max_dips':
                max_dips = extract_max_dips_based_on_maxs(dips)
            elif dip_detection_algorithm == 'threshold_dips':
                max_dips = extract_max_dips_based_on_threshold(y_values, threshold)
            elif dip_detection_algorithm == 'manual_dips':
                max_dips = kwargs.get('manual_dips')
                if not max_dips:
                    raise ValueError('No dips provided: manual_dips must hold values. See help or doc string')
            elif dip_detection_algorithm == 'lin_reg_dips':
                if kwargs.get('include_lin_reg', False) is False:
                    raise ValueError(f'To use {dip_detection_algorithm}, please enable include_lin_reg=True or'
                                     f'include_lin_reg=<float> to provide a threshold for the slopes.'
                                     f'See the documentation for more information (such as this function\'s docstring)')
                max_dips = \
                    [(int(filtered_segments[i]['end_point'][0]), int(filtered_segments[i + 1]['start_point'][0]))
                     for i in range(len(filtered_segments) - 1)]
                # Accommodate for two segments making one steady state
                max_dips = [dip for dip in max_dips if dip[0] != dip[1]]
            else:
                raise NotImplementedError(f"The dip detection algorithm '{dip_detection_algorithm}' is not implemented")

            # For a dip, get the maximal draw down (1- Robustness) Information and Recovery Information
            # Both infos are used later for adding the bars
            mdd_info = extract_mdd_from_dip(max_dips, y_values)
            recovery_info = get_recovery(y_values, max_dips, algorithm=recovery_algorithm)
            dip_auc_info = get_dip_auc(y_values, max_dips)

            # Draw the detected dips
            for max_dip, info in mdd_info.items():
                # Draw Recovery Time Line
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[max_dip[0], max_dip[1]],
                        y=[y_values[max_dip[0]], y_values[max_dip[0]]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                        name=f'Max Dips - {s.name}',
                        legendgroup=f'Max Dips - {s.name}'
                    )
                )
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[max_dip[1]],
                        y=[y_values[max_dip[0]]],
                        mode='markers',
                        marker=dict(symbol='x', color=fig.layout.template.layout.colorway[i]),
                        name=f'Max Dips - {s.name}',
                        legendgroup=f'Max Dips - {s.name}'
                    )
                )
                # Draw Maximal Drawdown
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[info['line'][0][0], info['line'][1][0]],
                        y=[info['line'][0][1], info['line'][1][1]],
                        mode='lines',
                        line=dict(color=fig.layout.template.layout.colorway[i], width=2, dash='dash'),
                        name=f'Max Drawdown {s.name}',
                        legendgroup=f'Max Dips - {s.name}'
                    )
                )

                if kwargs.get('include_dip_auc'):
                    # And make bars for the AUC of each dip
                    dip_auc_bars.append(
                        go.Bar(
                            x=[max_dip[1]],
                            y=[dip_auc_info[max_dip]],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i]),
                            opacity=0.5,
                            text="A ",
                            textangle=90,
                            textfont=BAR_TEXT_FONT,
                            textposition='outside',
                            name=f'(Local) AUC for dip {max_dip} - {s.name}',
                            hovertext=f'(Local) AUC for dip {max_dip} - {s.name}',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                            legendgroup=f'[T-Dip] (Local) AUC per dip - {s.name}',
                        )
                    )

            for _, recovery in recovery_info.items():
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[recovery['line'][0][0], recovery['line'][1][0]],
                        y=[recovery['line'][0][1], recovery['line'][1][1]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                        name=f'Recovery Level Line {s.name}',
                        legendgroup=f'Max Dips - {s.name}',
                    )
                )

            ###############################
            # [T-Dip] Core Resilience
            if kwargs.get('include_bars'):
                assert set(max_dips) == set(mdd_info.keys()), "Keys (Dips) do no match"
                for max_dip, info in mdd_info.items():
                    maximal_dips_bars.append(
                        go.Bar(
                            x=[info['line'][0][0]],
                            y=[mdd_to_robustness(info['value'])],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i]),
                            opacity=0.5,
                            text="R ",
                            textangle=90,
                            textfont=BAR_TEXT_FONT,
                            textposition='outside',
                            name=f'Robustness - {s.name}',
                            hovertext=f'Robustness - {s.name}',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                            legendgroup=f'[T-Dip] Resilience - {s.name}',
                        )
                    )
                    maximal_dips_bars.append(
                        go.Bar(
                            x=[max_dip[1]],
                            y=[dip_to_recovery_rate(max_dip)],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i]),
                            opacity=0.5,
                            text="RR",
                            textangle=90,
                            textfont=BAR_TEXT_FONT,
                            textposition='outside',
                            name=f'Recovery Rate - {s.name}',
                            hovertext=f'Recovery Rate - {s.name}',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                            legendgroup=f'[T-Dip] Resilience - {s.name}',
                        )
                    )
                for _, recovery in recovery_info.items():
                    maximal_dips_bars.append(
                        go.Bar(
                            x=[recovery['line'][0][0]],
                            y=[recovery['relative_recovery']],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i]),
                            opacity=0.5,
                            text="AC",
                            textangle=90,
                            textfont=BAR_TEXT_FONT,
                            textposition='outside',
                            name=f'Adaptive Capacity - {s.name}',
                            hovertext=f'Adaptive Capacity - {s.name}',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                            legendgroup=f'[T-Dip] Resilience - {s.name}',
                        )
                    )
            ###############
            # [T-Dip] Integrated Resilience Metric (IRM)
            if kwargs.get('include_irm'):
                irm = get_max_dip_integrated_resilience_metric(y_values, max_dips)
                assert set(max_dips) == set(irm.keys()), "Keys (Dips) do no match"
                for dip, irm_value in irm.items():
                    irm_bars.append(
                        go.Bar(
                            x=[dip[1]],
                            y=[irm_value],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i],
                                        line=dict(width=3,
                                                  color=fig.layout.template.layout.colorway[i]
                                                  )
                                        ),
                            opacity=0.5,
                            text="IRM",
                            textfont=BAR_TEXT_FONT,
                            textposition='outside',
                            textangle=90,
                            name=f'IRM - {s.name}',
                            hovertext=f'IRM {max_dip} - {s.name}',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                            legendgroup=f'[T-Dip] IRM - {s.name}',
                            yaxis='y6'
                        )
                    )
                    if not irm_value == 0:
                        continue
                    irm_bars.append(
                        go.Scatter(
                            x=[dip[1]],
                            y=[0],  # Match the zero value
                            mode='markers',
                            marker=dict(size=10, color='rgba(0,0,0,0)'),  # Invisible marker
                            hovertext=[f'IRM=0 {max_dip} - {s.name}'],  # Custom hovertext
                            hoverinfo='text',  # Only show hovertext
                            legendgroup=f'[T-Dip] IRM - {s.name}',
                            yaxis='y6',
                        )
                    )

            ##############################
            # [T-Dip] "antiFragility"
            if kwargs.get('calc_res_over_time'):
                # Construct input
                dips_resilience = {d: {} for d in max_dips}
                if kwargs.get('include_bars'):
                    assert set(dips_resilience.keys()) == set(mdd_info.keys()), "Keys (Dips) do no match"
                    for dip, mdd in mdd_info.items():
                        dips_resilience[dip]['robustness'] = mdd_to_robustness(mdd['value'])
                        dips_resilience[dip]['recovery'] = recovery_info[dip[1]]['relative_recovery']
                        dips_resilience[dip]['recovery rate'] = dip_to_recovery_rate(dip)

                if kwargs.get('include_dip_auc'):
                    assert set(dips_resilience.keys()) == set(dip_auc_info.keys()), "Keys (Dips) do no match"
                    for dip, auc in dip_auc_info.items():
                        dips_resilience[dip]['auc'] = auc

                if kwargs.get('include_irm'):
                    for dip, irm_value in irm.items():
                        dips_resilience[dip]['IRM'] = irm_value

                if kwargs.get('include_bars') or kwargs.get('include_dip_auc') + kwargs.get('include_irm') > 1:
                    for dip in max_dips:
                        dips_resilience[dip]['overall'] = np.mean(list(dips_resilience[dip].values()))

                antifragility = resilience_over_time(dips_resilience)

                # take output and draw the traces
                for metric, antifrag_dict in antifragility.items():
                    if antifrag_dict is None:  # metric had less than 2 instances
                        continue
                    name = f"Degree of Antifragility {f'under {metric}' if metric != 'overall' else '(overall)'}"
                    antifrag_diff_qu_traces.append(
                        go.Scatter(
                            x=[global_x_min - (1 + i), global_x_max + 1 + i],  # Extend the line to visualize better
                            y=[antifrag_dict.get('alpha_u'), antifrag_dict.get('alpha_u')],
                            mode="markers+lines",
                            marker=dict(size=10, symbol="cross", line=dict(width=1, color="DarkSlateGrey"))
                            if not metric == "overall"
                            else dict(size=14, symbol="diamond", line=dict(width=2, color="DarkSlateGrey")),
                            name=f'{name} - {s.name}',
                            hovertext=f'{name} - <br>{s.name}',
                            legendgroup=f'Degree of Antifragility - {s.name}',
                            line=dict(dash='dash', color=fig.layout.template.layout.colorway[i]),
                            yaxis='y5',
                            hoverinfo='x+y+text',  # Show x, y, and hover text
                        )
                    )

    # Include threshold line if requested
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh') \
            or dip_detection_algorithm == "threshold_dips":
        threshold_line.append(go.Scatter(
            x=[global_x_min, global_x_max],  # Extend the line across the global x-axis range
            y=[threshold / 100, threshold / 100],
            mode='lines',
            name=f'Threshold: {threshold}%',
            line=dict(dash='dash', color='red')
        ))

    # Update layout to include secondary y-axes
    fig.update_layout(
        yaxis2=dict(
            title='Time below Threshold',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Dip below Threshold Count',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True
        ),
        yaxis4=dict(
            title='Drawdown (%)',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True
        ),
        yaxis5=dict(
            title='Degree of Antifragility alpha_u',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True,
            zeroline=True,
            rangemode="tozero",
        ),
        yaxis6=dict(
            title='Integrated Resilience Metric (IRM)',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True,
            zeroline=True
        )
    )

    # Combine all requested traces
    all_traces = list(series)
    if kwargs.get('include_auc'):
        all_traces += auc_traces
    if kwargs.get('include_count_below_thresh'):
        all_traces += count_below_thresh_traces
    if kwargs.get('include_time_below_thresh'):
        all_traces += time_below_thresh_traces
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh') \
            or dip_detection_algorithm == "threshold_dips":
        all_traces += threshold_line
    if kwargs.get('include_draw_downs_traces'):
        all_traces += drawdown_traces
    if kwargs.get('include_smooth_criminals'):
        all_traces += smooth_criminals
    if kwargs.get('include_dips'):
        all_traces += dips_horizontal_shapes
    if kwargs.get('include_draw_downs_shapes'):
        all_traces += draw_down_shapes
    if dip_detection_algorithm:  # may be set automatically therefore not via kwarg
        all_traces += maximal_dips_shapes
    if kwargs.get('include_bars'):
        all_traces += maximal_dips_bars
    if kwargs.get('include_derivatives'):
        all_traces += derivative_traces
    if kwargs.get('include_lin_reg'):
        all_traces += lin_reg_traces
    if kwargs.get('include_dip_auc'):
        all_traces += dip_auc_bars
    if kwargs.get('include_irm'):
        all_traces += irm_bars
    if kwargs.get('calc_res_over_time'):
        all_traces += antifrag_diff_qu_traces

    # Update the figure with new data and layout
    fig = go.Figure(data=all_traces, layout=fig.layout)
    fig.update_layout(
        barmode='overlay',
        margin=dict(l=10, r=20, t=10, b=10),
        legend=dict(
            x=.02,
            y=.98,
            xanchor='left',
            yanchor='top'
        ),
        updatemenus=[{
            'x': 0.0,   # Horizontal position (fraction of plot width)
            'y': 1.0,   # Vertical position (fraction of plot height)
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'name': 'Toggle legend',
                'label': '≡',
                'method': 'relayout',
                'args': [{'showlegend': True}],
                'args2': [{'showlegend': False}]
            }],
            'bgcolor': 'rgba(255, 255, 255, 0.3)',  # 70% transparent background
        }]
    )

    # Construct the legend
    legend_groups_seen = set()
    for trace in fig.data:
        legend_group = trace.legendgroup
        if legend_group:
            trace.update(showlegend=False)
            if legend_group not in legend_groups_seen:
                # Extract color based on trace type
                if hasattr(trace, 'line') and hasattr(trace.line, 'color') and trace.line.color is not None:
                    color = trace.line.color
                elif hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and trace.marker.color is not None:
                    color = trace.marker.color
                elif hasattr(trace, 'fillcolor'):
                    color = trace.fillcolor
                else:
                    color = None

                # Create a dummy trace for the legend
                dummy_trace = go.Scatter(
                    x=[None],  # Dummy data
                    y=[None],  # Dummy data
                    mode='markers',
                    showlegend=True,
                    name=legend_group,  # Use the legend group name
                    legendgroup=legend_group,
                    line=dict(color=color) if color else None,  # Preserve color if available
                    marker=dict(color=color, size=7) if color else dict(size=7)
                )
                # Add the dummy trace to the figure
                fig.add_trace(dummy_trace)

                # Add the legend group to the seen set
                legend_groups_seen.add(legend_group)
        else:
            trace.update(showlegend=True)

    return fig
