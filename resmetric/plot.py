from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pwlf
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
    get_max_dip_auc,
    mdd_to_robustness,
    dip_to_recovery_rate,
    get_max_dip_integrated_resilience_metric,
    extract_max_dips_based_on_threshold
)


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

        - include_auc (bool): Include AUC-related traces. (AUC devided by the length of the time frame and
          different kernels applied)
        - include_max_dip_auc (bool): Include AUC bars for the AUC of one maximal dip
          (AUC devided by the length of the time frame)
        - include_count_below_thresh (bool): Include traces counting dips below
          the threshold.
        - include_time_below_thresh (bool): Include traces accumulating time
          below the threshold.
        - threshold (float): Threshold for count and time traces (default is 80).
        - include_draw_downs_traces (bool): Include traces representing the
          relative loss at each point in the time series, calculated as the
          difference between the current value and the highest value reached
          up to that point, divided by that highest value.
        - include_smooth_criminals (bool): Include smoothed series.
          (Hee-Hee! Ow!)
        - include_dips (bool): Include detected dips.
        - include_draw_downs_shapes (bool): Include shapes of local draw-downs.
        - include_maximal_dips (bool): Include maximal dips, maximal draw-downs,
          and recoveries.
        - include_bars (bool): Include bars for robustness, recovery and recovery time.
        - recovery_algorithm (str or None): Decides the recovery algorithm. Can either be 'adaptive_capacity' (default)
          or 'recovery_ability'. The first one is the ratio of new to prior steady state's value (Q(t_ns) / Q(t_0)).
          The last one is abs((Q(t_ns) - Q(t_r))/ (Q(t_0) - Q(t_r)))
          where Q(t_r) is the local minimum within a dip (Robustness).
        - include_gr (bool): Include the Integrated Resilience Metric
          (cf. Sansavini, https://doi.org/10.1007/978-94-024-1123-2_6, Chapter 6, formula 6.12).
          Requires kwarg recovery_algorithm='recovery_ability'.
        - include_derivatives (bool): Include derivatives traces.
        - include_lin_reg (bool): Include linear regression traces.
        - penalty_factor (float): Penalty factor for Bayesian Optimization
          (default is 0.05).
        - dimensions (int): Dimensions for Bayesian Optimization (default is 10)
          (Max. N. of segments for lin. reg.).
        - weighted_auc_half_life (float): Half-life for weighted AUC calculation
          (default is 2).
        - smoother_threshold (int): Threshold for smoother function in percent
          (default is 2[%]).
        - calc_res_over_time (bool): Calculate the differential quotient for every Core Resilience-Related Trace.
        - dip_detection_algorithm (str or None): Specifies the dip detection algorithm to use.
          It can be 'max_dips' (default), 'threshold_dips', 'manual_dips', or None.
        - manual_dips (list of tuples or None): If 'manual_dips' is selected as the dip detection algorithm,
          this should be a list of tuples specifying the manual dips.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly Figure object with the specified traces and analyses included.
    """
    # Set default dip detection algorithm to 'max_dips'
    dip_detection_algorithm = kwargs.get('dip_detection_algorithm', 'max_dips')
    recovery_algorithm = kwargs.get('recovery_algorithm', 'adaptive_capacity')

    # # Validate the include_gr parameter
    # if kwargs.get('include_gr') and recovery_algorithm != 'recovery_ability':
    #     raise ValueError(
    #         "The 'include_gr' option requires the 'recovery_algorithm' to be set to 'recovery_ability'. "
    #         "Please set 'recovery_algorithm' to 'recovery_ability' to include the Integrated Resilience Metric."
    #     )

    # TODO [END] update readme + add the new anti-fragility func
    # TODO [END] update ASCII art

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
    max_dip_auc_bars = []
    gr_bars = []

    # Retrieve optional arguments with defaults
    threshold = kwargs.get('threshold', 80)
    penalty_factor = kwargs.get('penalty_factor', 0.05)
    dimensions = kwargs.get('dimensions', 10)
    weighted_auc_half_life = kwargs.get('weighted_auc_half_life', 2)
    smoother_threshold = kwargs.get('smoother_threshold', 2)

    # Initialize variables for global x limits
    global_x_min = float('inf')
    global_x_max = float('-inf')

    for i, s in enumerate(series):
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
            #TODO implement lr dip

        # For a dip, get the maximal draw down (1- Robustness) Information and Recovery Information
        # Both infos are used later for adding the bars
        mdd_info = extract_mdd_from_dip(max_dips, y_values)
        recovery_info = get_recovery(y_values, max_dips, algorithm=recovery_algorithm)
        max_dip_auc_info = get_max_dip_auc(y_values, max_dips)

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

            if kwargs.get('include_max_dip_auc'):
                # And make bars for the AUC of each dip
                max_dip_auc_bars.append(
                    go.Bar(
                        x=[max_dip[1]],
                        y=[max_dip_auc_info[max_dip]],
                        width=1,
                        marker=dict(color=fig.layout.template.layout.colorway[i]),
                        opacity=0.25,
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
                    name=f'Recovery Line {s.name}',
                    legendgroup=f'Max Dips - {s.name}',
                )
            )

        # [T-Dip] Fit the piecewise linear model and add to traces if requested
        if kwargs.get('include_lin_reg'):
            # Suppress only UserWarnings temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                # Perform Bayesian Optimization to find the optimal number of segments
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating linear regression"
                      f" of series {i + 1} of {len(series)}")
                optimal_segments = _perform_bayesian_optimization(x_values, y_values,
                                                                  penalty_factor=penalty_factor, dimensions=dimensions)
                pwlf_model = pwlf.PiecewiseLinFit(x_values, y_values)
                pwlf_model.fit(optimal_segments)
                # y_hat = pwlf_model.predict(x_values)
                # y_hat = ""

                # if i == 0:
                # breakpoints = [0., 5.4322327, 57.26332236, 58.45461669, 85.81890165, 101.53587274, 101.80619173,
                #                135.]
                # y_hat = "0.85191131 0.76847839 0.68504547 0.60161254 0.51817962 0.434746 0.39885111 0.399145 0.39943888 0.39973276 0.40002664 0.4003205 0.4006144  0.40090829 0.40120217 0.40149605 0.40178993 0.40208381 0.40237769 0.40267158 0.40296546 0.40325934 0.40355322 0.4038471 0.40414098 0.40443486 0.40472875 0.40502263 0.40531651 0.40561039 0.40590427 0.40619815 0.40649204 0.40678592 0.4070798  0.40737368 0.40766756 0.40796144 0.40825532 0.40854921 0.40884309 0.40913697 0.40943085 0.40972473 0.41001861 0.4103125  0.41060638 0.4109002 0.41119414 0.41148802 0.4117819  0.41207578 0.41236967 0.41266355 0.41295743 0.41325131 0.41354519 0.41383907 0.31021241 0.2467688 0.24778458 0.24880035 0.24981612 0.25083189 0.25184767 0.25286344 0.25387921 0.25489498 0.25591075 0.25692653 0.2579423  0.25895807 0.25997384 0.26098962 0.26200539 0.26302116 0.26403693 0.2650527 0.26606848 0.26708425 0.26810002 0.26911579 0.27013156 0.27114734 0.27216311 0.27317888 0.27182451 0.2597527  0.24768089 0.23560908 0.22353727 0.21146546 0.19939365 0.18732184 0.17525003 0.16317821 0.1511064  0.13903459 0.12696278 0.11489097 0.10281916 0.09074735 0.37601681 0.37518869 0.37436058 0.37353247 0.37270435 0.37187624 0.37104813 0.37022002 0.3693919  0.36856379 0.36773568 0.36690756 0.36607945 0.36525134 0.36442322 0.36359511 0.362767   0.36193888 0.36111077 0.36028266 0.35945455 0.35862643 0.35779832 0.35697021 0.35614209 0.35531398 0.35448587 0.35365775 0.35282964 0.35200153 0.35117341 0.3503453  0.34951719 0.34868908"
                # slopes = [-8.34329234e-02,  2.93881551e-04, -1.40772633e-01,  1.01577219e-03, -1.20718111e-02,  1.07983132e+00, -8.28113063e-04]
                # elif i == 1:
                #     breakpoints = [0., 6.90296868, 55.74162991, 57.60297693, 95., 100.889095, 102.09569763, 103.129211,
                #                    135.]
                #     y_hat = "0.72534864 0.75215664 0.77896464 0.80577264 0.83258065 0.85938865 0.88619665 0.91042743 0.91067467 0.91092191 0.91116915 0.91141639 0.91166363 0.91191087 0.91215811 0.91240535 0.9126526  0.91289984 0.91314708 0.91339432 0.91364156 0.9138888  0.91413604 0.91438328 0.91463052 0.91487776 0.915125   0.91537224 0.91561948 0.91586672 0.91611396 0.9163612  0.91660845 0.91685569 0.91710293 0.91735017 0.91759741 0.91784465 0.91809189 0.91833913 0.91858637 0.91883361 0.91908085 0.91932809 0.91957533 0.91982257 0.92006981 0.92031705 0.9205643  0.92081154 0.92105878 0.92130602 0.92155326 0.9218005 0.92204774 0.92229498 0.90006571 0.81331949 0.76059833 0.75955256 0.7585068  0.75746103 0.75641527 0.75536951 0.75432374 0.75327798 0.75223221 0.75118645 0.75014069 0.74909492 0.74804916 0.74700339 0.74595763 0.74491187 0.7438661  0.74282034 0.74177457 0.74072881 0.73968305 0.73863728 0.73759152 0.73654575 0.73549999 0.73445423 0.73340846 0.7323627  0.73131693 0.73027117 0.72922541 0.72817964 0.72713388 0.72608812 0.72504235 0.72399659 0.72295082 0.72190506 0.78370612 0.84550719 0.90730825 0.96910932 1.03091038 1.00042938 0.23014837 0.4685917  0.51407358 0.51508341 0.51609325 0.51710308 0.51811292 0.51912275 0.52013258 0.52114242 0.52215225 0.52316209 0.52417192 0.52518175 0.52619159 0.52720142 0.52821126 0.52922109 0.53023092 0.53124076 0.53225059 0.53326043 0.53427026 0.53528009 0.53628993 0.53729976 0.5383096  0.53931943 0.54032927 0.5413391 0.54234893 0.54335877 0.5443686 0.54537844"
                #     slopes = [2.68080029e-02,  2.47240621e-04, -8.67462224e-02, -1.04576396e-03, 6.18010647e-02, -7.70281016e-01,  3.45191395e-01,  1.00983408e-03]
                # y_hat = y_hat.split()

                # lin_reg_traces.append(
                #     go.Scatter(
                #         x=x_values,
                #         y=y_hat,
                #         mode='lines',
                #         line=dict(color=fig.layout.template.layout.colorway[i]),
                #         # name=f'PWLF ({optimal_segments} Segments) - {s.name}',  TODO
                #         name=f'PWLF (x Segments) - {s.name}',
                #         legendgroup=f'PWLF - {s.name}'
                #     )
                # )

                # Extract breakpoints and slopes
                breakpoints = pwlf_model.fit_breaks

                slopes = pwlf_model.calc_slopes()
                for j in range(len(breakpoints)):
                    try:
                        next_bigger = round(breakpoints[j+1])
                    except:
                        next_bigger = -1
                    this = round(breakpoints[j])
                    breakpoints[j] = this if not next_bigger == this else this-1
                # print(f"bp:{breakpoints}")

                # Extract start and end points of each segment
                segments = []
                for j in range(len(breakpoints) - 1):
                    start_x = breakpoints[j]
                    end_x = breakpoints[j + 1]
                    start_y = pwlf_model.predict(start_x)[0]
                    end_y = pwlf_model.predict(end_x)[0]  #  TODO
                    # start_y = y_hat[start_x]
                    # end_y = y_hat[end_x]
                    slope = slopes[j]
                    segments.append({
                        'start_point': (start_x, start_y),
                        'end_point': (end_x, end_y),
                        'slope': slope
                    })

                filtered_segments = [seg for seg in segments if abs(seg['slope']) < .5e-2]

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
                            legendgroup=f'PWLF - {s.name}'
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
                        opacity=0.25,
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
                        opacity=0.25,
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
                        opacity=0.25,
                        name=f'Rel. Recovery - {s.name}',
                        hovertext=f'Rel. Recovery - {s.name}',
                        hoverinfo='x+y+text',  # Show x, y, and hover text
                        legendgroup=f'[T-Dip] Resilience - {s.name}',
                    )
                )
        ###############
        # [T-Dip] Integrated Resilience Metric (IRM) GR
        if kwargs.get('include_gr'):
            gr = get_max_dip_integrated_resilience_metric(y_values, max_dips)
            assert set(max_dips) == set(gr.keys()), "Keys (Dips) do no match"
            for dip, gr_value in gr.items():
                gr_bars.append(
                    go.Bar(
                        x=[dip[1]],
                        y=[gr_value],
                        width=1,
                        marker=dict(color=fig.layout.template.layout.colorway[i]),
                        opacity=0.25,
                        name=f'IRM GR - {s.name}',
                        hovertext=f'IRM GR {max_dip} - {s.name}',
                        hoverinfo='x+y+text',  # Show x, y, and hover text
                        legendgroup=f'[T-Dip] IRM GR - {s.name}',
                        yaxis='y6'
                    )
                )

        ##############################
        # [T-Dip] "antiFragility"
        # mdd_info = extract_mdd_from_dip(max_dips, y_values) # Robustness
        #  recovery_info = get_recovery(y_values, max_dips)
        # AUC
        # length
        if (kwargs.get('calc_res_over_time') and
                (kwargs.get('include_bars') or kwargs.get('include_dip_auc') or kwargs.get('include_gr'))):
            # Construct input
            dips_resilience = {d: {} for d in max_dips}
            if kwargs.get('include_bars'):
                assert set(dips_resilience.keys()) == set(mdd_info.keys()), "Keys (Dips) do no match"
                for dip, mdd in mdd_info.items():
                    dips_resilience[dip]['robustness'] = mdd_to_robustness(mdd['value'])
                    dips_resilience[dip]['recovery'] = recovery_info[dip[1]]['relative_recovery']
                    dips_resilience[dip]['recovery rate'] = dip_to_recovery_rate(dip)

            if kwargs.get('include_max_dip_auc'):
                assert set(dips_resilience.keys()) == set(max_dip_auc_info.keys()), "Keys (Dips) do no match"
                for dip, auc in max_dip_auc_info.items():
                    dips_resilience[dip]['auc'] = auc

            if kwargs.get('include_gr'):
                for dip, gr_value in gr.items():
                    dips_resilience[dip]['IRM GR'] = gr_value

            # take output and draw the traces
            for metric, metric_change in resilience_over_time(dips_resilience).items():
                antifrag_diff_qu_traces.append(
                    go.Scatter(
                        x=[end for end, _ in metric_change.get('diff_q')],
                        y=[quotient for _, quotient in metric_change.get('diff_q')],
                        mode='lines+markers',
                        line=dict(color=fig.layout.template.layout.colorway[i],
                                  dash='dot'),
                        marker=dict(
                            symbol='cross',
                            size=8,
                            color='black'
                        ),
                        name=f'Diff. quot. of {metric} - {s.name}',
                        hovertext=f'Diff. quot. of {metric} - <br>{s.name}',
                        legendgroup=f'"Antifragility" - {s.name}',
                        hoverinfo='x+y+text',  # Show x, y, and hover text
                        yaxis='y5'
                    )
                )
                antifrag_diff_qu_traces.append(
                    go.Scatter(
                        x=[global_x_min, global_x_max],  # Extend the line across the global x-axis range
                        y=[metric_change.get('overall'), metric_change.get('overall')],
                        mode='lines',
                        name=f'Mean Diff. quot. of {metric} - {s.name}',
                        hovertext=f'Mean Diff. quot. of {metric} - <br>{s.name}',
                        legendgroup=f'"Antifragility" - {s.name}',
                        line=dict(dash='dash', color='black'),
                        yaxis='y5',
                        hoverinfo='x+y+text',  # Show x, y, and hover text
                    )
                )

        # Update the original series with a pale color
        s.update(
            mode='lines+markers',
            marker=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i])),
            line=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i]))
        )
        if i == 0:
            pass
            # break  # TODO Remove

    # Include threshold line if requested
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh'):
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
            title='Differential Quotient "Antifragility"',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True,
            zeroline=True
        ),
        yaxis6=dict(
            title='Integrated Resilience Metric (IRM) GR',
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
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh'):
        all_traces += threshold_line
    if kwargs.get('include_draw_downs_traces'):
        all_traces += drawdown_traces
    if kwargs.get('include_smooth_criminals'):
        all_traces += smooth_criminals
    if kwargs.get('include_dips'):
        all_traces += dips_horizontal_shapes
    if kwargs.get('include_draw_downs_shapes'):
        all_traces += draw_down_shapes
    if kwargs.get('include_maximal_dips'):
        all_traces += maximal_dips_shapes
    if kwargs.get('include_bars'):
        all_traces += maximal_dips_bars
    if kwargs.get('include_derivatives'):
        all_traces += derivative_traces
    if kwargs.get('include_lin_reg'):
        all_traces += lin_reg_traces
    if kwargs.get('include_max_dip_auc'):
        all_traces += max_dip_auc_bars
    if kwargs.get('include_gr'):
        all_traces += gr_bars
    if kwargs.get('calc_res_over_time'):
        all_traces += antifrag_diff_qu_traces

    # Update the figure with new data and layout
    fig = go.Figure(data=all_traces, layout=fig.layout)
    fig.update_layout(barmode='overlay', margin=dict(l=10, r=20, t=10, b=10), legend=dict(
        x=.02,
        y=.98,
        xanchor='left',
        yanchor='top'
    ))

    # Construct the legend
    legend_groups_seen = set()
    for trace in fig.data:
        legend_group = trace.legendgroup
        if legend_group:
            trace.update(showlegend=False)
            if legend_group not in legend_groups_seen:
                # Extract color based on trace type
                if hasattr(trace, 'line') and hasattr(trace.line, 'color'):
                    color = trace.line.color
                elif hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
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
                    marker=dict(size=0),  # Make it invisible
                    showlegend=True,
                    name=legend_group,  # Use the legend group name
                    legendgroup=legend_group,
                    line=dict(color=color) if color else None  # Preserve line color if applicable
                )
                # Add the dummy trace to the figure
                fig.add_trace(dummy_trace)

                # Add the legend group to the seen set
                legend_groups_seen.add(legend_group)
        else:
            trace.update(showlegend=True)

    return fig
